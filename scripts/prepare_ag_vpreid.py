#!/usr/bin/env python3
"""Convert AG-VPReID frame folders into OpenGait RGB pickle sequences."""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image


CAMERA_PATTERN = re.compile(r"C([0-5])", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        action="append",
        required=True,
        type=Path,
        help="Raw train directory shaped <root>/<person-id>/<tracklet>/<frames>; repeatable.",
    )
    parser.add_argument(
        "--test-root",
        action="append",
        required=True,
        type=Path,
        help="Raw test directory with the same layout; repeatable.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--partition-out",
        type=Path,
        default=Path("datasets/AG-VPReID/AG_VPReID.json"),
    )
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rgb.pkl files. By default, completed tracklets are reused.",
    )
    return parser.parse_args()


def camera_name(tracklet_dir, frame_paths):
    candidates = [frame_paths[0].name, tracklet_dir.name]
    for candidate in candidates:
        match = CAMERA_PATTERN.search(candidate)
        if match:
            return f"C{match.group(1)}"
    raise ValueError(
        f"Could not infer a C0-C5 camera ID from tracklet {tracklet_dir}. "
        "The official frame name should contain a token such as C4."
    )


def load_tracklet(frame_paths, height, width):
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            frame = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        frames.append(frame)
    return np.stack(frames, axis=0)


def identity_dirs(roots):
    result = {}
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        for path in sorted(root.iterdir()):
            if path.is_dir():
                result.setdefault(path.name, []).append(path)
    return result


def convert_split(split_name, roots, output_root, height, width, overwrite):
    identities = identity_dirs(roots)
    completed_ids = []
    tracklet_count = 0
    frame_count = 0

    for person_id, person_dirs in sorted(identities.items()):
        person_has_data = False
        for person_dir in person_dirs:
            for tracklet_dir in sorted(path for path in person_dir.iterdir() if path.is_dir()):
                frame_paths = sorted(
                    path
                    for path in tracklet_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                if not frame_paths:
                    continue
                camera = camera_name(tracklet_dir, frame_paths)
                destination = output_root / person_id / camera / tracklet_dir.name / "rgb.pkl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not destination.exists():
                    sequence = load_tracklet(frame_paths, height, width)
                    temporary = destination.with_suffix(".pkl.tmp")
                    with temporary.open("wb") as handle:
                        pickle.dump(sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    temporary.replace(destination)
                person_has_data = True
                tracklet_count += 1
                frame_count += len(frame_paths)
        if person_has_data:
            completed_ids.append(person_id)

    print(
        f"{split_name}: {len(completed_ids)} identities, "
        f"{tracklet_count} tracklets, {frame_count} frames"
    )
    return completed_ids


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive integers")
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_ids = convert_split(
        "train",
        args.train_root,
        args.output_root,
        args.height,
        args.width,
        args.overwrite,
    )
    test_ids = convert_split(
        "test",
        args.test_root,
        args.output_root,
        args.height,
        args.width,
        args.overwrite,
    )
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(
            "Train/test identity overlap detected. Check the supplied roots. "
            f"Examples: {overlap[:5]}"
        )

    partition = {"TRAIN_SET": sorted(train_ids), "TEST_SET": sorted(test_ids)}
    args.partition_out.parent.mkdir(parents=True, exist_ok=True)
    with args.partition_out.open("w", encoding="utf-8") as handle:
        json.dump(partition, handle, indent=2)
        handle.write("\n")
    print(f"Partition written to {args.partition_out}")
    print(f"Set model_cfg.SeparateBNNecks.class_num to {len(train_ids)}")


if __name__ == "__main__":
    main()

