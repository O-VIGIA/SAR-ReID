"""Sequence-consistent RGB transforms used by STAR-CVI.

The transforms accept either ``[T, H, W, C]`` or ``[T, C, H, W]`` NumPy
arrays. Geometric and photometric parameters are shared across a tracklet by
default so temporal correspondence is not destroyed by augmentation.
"""

import math
import random

import cv2
import numpy as np


def _to_thwc(seq):
    array = np.asarray(seq)
    if array.ndim != 4:
        raise ValueError(f"Expected a 4-D RGB sequence, received shape {array.shape}")
    if array.shape[-1] in (1, 3):
        return array, "thwc"
    if array.shape[1] in (1, 3):
        return array.transpose(0, 2, 3, 1), "tchw"
    raise ValueError(
        "Cannot identify the RGB channel in sequence shaped "
        f"{array.shape}; expected [T,H,W,C] or [T,C,H,W]."
    )


def _restore_layout(array, layout):
    if layout == "tchw":
        return array.transpose(0, 3, 1, 2)
    return array


def _restore_dtype(array, dtype, upper):
    array = np.clip(array, 0.0, upper)
    if np.issubdtype(dtype, np.integer):
        return np.rint(array).astype(dtype)
    return array.astype(dtype, copy=False)


def _warp_frames(frames, matrix, perspective=False):
    height, width = frames.shape[1:3]
    warped = []
    for frame in frames:
        if perspective:
            out = cv2.warpPerspective(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        else:
            out = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        if out.ndim == 2:
            out = out[..., None]
        warped.append(out)
    return np.stack(warped, axis=0)


class SequenceAwareColorJitter:
    def __init__(self, prob=0.5, brightness=0.2, contrast=0.2, saturation=0.2):
        self.prob = float(prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)

    def __call__(self, seq):
        if random.random() >= self.prob:
            return seq
        frames, layout = _to_thwc(seq)
        dtype = frames.dtype
        upper = 1.0 if np.issubdtype(dtype, np.floating) and frames.max(initial=0) <= 1.5 else 255.0
        out = frames.astype(np.float32, copy=True)

        brightness = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        contrast = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        saturation = random.uniform(1.0 - self.saturation, 1.0 + self.saturation)

        out *= brightness
        channel_mean = out.mean(axis=(1, 2), keepdims=True)
        out = channel_mean + contrast * (out - channel_mean)
        if out.shape[-1] == 3:
            gray = (
                0.299 * out[..., 0:1]
                + 0.587 * out[..., 1:2]
                + 0.114 * out[..., 2:3]
            )
            out = gray + saturation * (out - gray)

        out = _restore_dtype(out, dtype, upper)
        return _restore_layout(out, layout)


class SequenceAwareBoxJitter:
    def __init__(
        self,
        prob=0.5,
        max_shift_ratio=0.06,
        max_scale_ratio=0.08,
        per_frame_jitter_ratio=0.0,
    ):
        self.prob = float(prob)
        self.max_shift_ratio = float(max_shift_ratio)
        self.max_scale_ratio = float(max_scale_ratio)
        self.per_frame_jitter_ratio = float(per_frame_jitter_ratio)

    def __call__(self, seq):
        if random.random() >= self.prob:
            return seq
        frames, layout = _to_thwc(seq)
        height, width = frames.shape[1:3]
        scale = random.uniform(1.0 - self.max_scale_ratio, 1.0 + self.max_scale_ratio)
        shift_x = random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * width
        shift_y = random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * height
        base = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, scale)
        base[0, 2] += shift_x
        base[1, 2] += shift_y

        if self.per_frame_jitter_ratio <= 0:
            out = _warp_frames(frames, base)
        else:
            out = []
            for frame in frames:
                matrix = base.copy()
                matrix[0, 2] += random.uniform(-1.0, 1.0) * self.per_frame_jitter_ratio * width
                matrix[1, 2] += random.uniform(-1.0, 1.0) * self.per_frame_jitter_ratio * height
                out.append(_warp_frames(frame[None], matrix)[0])
            out = np.stack(out, axis=0)
        return _restore_layout(out.astype(frames.dtype, copy=False), layout)


class SequenceAwarePerspectiveAffine:
    def __init__(
        self,
        prob=0.3,
        max_perspective_ratio=0.05,
        max_rotate_degree=6.0,
        max_affine_shift_ratio=0.04,
    ):
        self.prob = float(prob)
        self.max_perspective_ratio = float(max_perspective_ratio)
        self.max_rotate_degree = float(max_rotate_degree)
        self.max_affine_shift_ratio = float(max_affine_shift_ratio)

    def __call__(self, seq):
        if random.random() >= self.prob:
            return seq
        frames, layout = _to_thwc(seq)
        height, width = frames.shape[1:3]
        source = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        jitter = np.float32(
            [
                [random.uniform(-1, 1) * width, random.uniform(-1, 1) * height]
                for _ in range(4)
            ]
        )
        target = source + self.max_perspective_ratio * jitter
        perspective = cv2.getPerspectiveTransform(source, target)

        angle = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        affine[0, 2] += random.uniform(-1.0, 1.0) * self.max_affine_shift_ratio * width
        affine[1, 2] += random.uniform(-1.0, 1.0) * self.max_affine_shift_ratio * height
        affine_3x3 = np.vstack([affine, [0.0, 0.0, 1.0]])
        matrix = affine_3x3 @ perspective
        out = _warp_frames(frames, matrix, perspective=True)
        return _restore_layout(out.astype(frames.dtype, copy=False), layout)


class SequenceAwareRandomErasing:
    def __init__(
        self,
        prob=0.35,
        sl=0.03,
        sh=0.1,
        r1=0.3,
        per_frame=False,
        per_frame_jitter_ratio=0.02,
        value=0.0,
    ):
        self.prob = float(prob)
        self.sl = float(sl)
        self.sh = float(sh)
        self.r1 = float(r1)
        self.per_frame = bool(per_frame)
        self.per_frame_jitter_ratio = float(per_frame_jitter_ratio)
        self.value = value

    def __call__(self, seq):
        if random.random() >= self.prob:
            return seq
        frames, layout = _to_thwc(seq)
        out = frames.copy()
        height, width = out.shape[1:3]
        area = height * width

        box = None
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect = random.uniform(self.r1, 1.0 / self.r1)
            erase_h = int(round(math.sqrt(target_area * aspect)))
            erase_w = int(round(math.sqrt(target_area / aspect)))
            if 0 < erase_h < height and 0 < erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                box = (top, left, erase_h, erase_w)
                break
        if box is None:
            return seq

        top, left, erase_h, erase_w = box
        for index in range(len(out)):
            if self.per_frame:
                current_top = random.randint(0, height - erase_h)
                current_left = random.randint(0, width - erase_w)
            else:
                dy = int(round(random.uniform(-1, 1) * self.per_frame_jitter_ratio * height))
                dx = int(round(random.uniform(-1, 1) * self.per_frame_jitter_ratio * width))
                current_top = min(max(0, top + dy), height - erase_h)
                current_left = min(max(0, left + dx), width - erase_w)
            out[index, current_top : current_top + erase_h, current_left : current_left + erase_w] = self.value
        return _restore_layout(out, layout)


class SequenceAwareHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def __call__(self, seq):
        if random.random() >= self.prob:
            return seq
        frames, layout = _to_thwc(seq)
        out = frames[:, :, ::-1, :].copy()
        return _restore_layout(out, layout)


class ClipRgbTransform:
    """Convert RGB sequences to ``[T,C,H,W]`` and apply CLIP normalization."""

    def __init__(self, mean=None, std=None):
        self.mean = np.asarray(
            mean or [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        self.std = np.asarray(
            std or [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
        ).reshape(1, 3, 1, 1)

    def __call__(self, seq):
        frames, _ = _to_thwc(seq)
        if frames.shape[-1] != 3:
            frames = np.repeat(frames, 3, axis=-1)
        frames = frames.astype(np.float32)
        if frames.max(initial=0) > 1.5:
            frames /= 255.0
        frames = frames.transpose(0, 3, 1, 2)
        return (frames - self.mean) / self.std


TRANSFORM_REGISTRY = {
    "SequenceAwareColorJitter": SequenceAwareColorJitter,
    "SequenceAwareBoxJitter": SequenceAwareBoxJitter,
    "SequenceAwarePerspectiveAffine": SequenceAwarePerspectiveAffine,
    "SequenceAwareRandomErasing": SequenceAwareRandomErasing,
    "SequenceAwareHorizontalFlip": SequenceAwareHorizontalFlip,
    "ClipRgbTransform": ClipRgbTransform,
}

