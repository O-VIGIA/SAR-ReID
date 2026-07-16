"""AG-VPReID retrieval protocols for OpenGait.

The official dataset defines C0-C3 as ground-view cameras and C4-C5 as
aerial-view cameras. The evaluator reports aerial-to-ground, ground-to-aerial,
ground-to-ground, and aerial-to-aerial retrieval with the standard person-ReID
same-identity/same-camera exclusion.
"""

import numpy as np

from evaluation.metric import cuda_dist, evaluate_many
from utils import get_msg_mgr


DEFAULT_GROUND_PLATFORMS = ("C0", "C1", "C2", "C3")
DEFAULT_AERIAL_PLATFORMS = ("C4", "C5")


def _camera_token(value):
    return str(value).split("-", 1)[0].strip().upper()


def _ensure_part_dimension(features):
    features = np.asarray(features)
    if features.ndim == 2:
        features = features[..., None]
    if features.ndim != 3:
        raise ValueError(
            "Expected retrieval embeddings shaped [N,C,P] or [N,C], "
            f"received {features.shape}."
        )
    return features


def _evaluate_protocol(features, labels, cameras, query_mask, gallery_mask, metric):
    query_features = features[query_mask]
    gallery_features = features[gallery_mask]
    query_labels = labels[query_mask]
    gallery_labels = labels[gallery_mask]
    query_cameras = cameras[query_mask]
    gallery_cameras = cameras[gallery_mask]

    if len(query_features) == 0 or len(gallery_features) == 0:
        raise ValueError(
            "The requested AG-VPReID protocol has an empty query or gallery. "
            "Check camera names and evaluator_cfg platform mappings."
        )

    distance = cuda_dist(query_features, gallery_features, metric).cpu().numpy()
    cmc, mean_ap, mean_inp = evaluate_many(
        distance,
        query_labels,
        gallery_labels,
        query_cameras,
        gallery_cameras,
    )
    return {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(mean_ap * 100.0),
        "mINP": float(mean_inp * 100.0),
    }


def evaluate_UAV_Ground(
    data,
    dataset,
    metric="euc",
    aerial_platforms=None,
    ground_platforms=None,
):
    """Evaluate the four aerial/ground protocols used by STAR-CVI."""
    del dataset  # The camera mapping is explicit and configurable below.
    features = _ensure_part_dimension(data["embeddings"])
    labels = np.asarray(data["labels"])
    cameras = np.asarray([_camera_token(value) for value in data["types"]])

    aerial_platforms = tuple(
        str(value).upper() for value in (aerial_platforms or DEFAULT_AERIAL_PLATFORMS)
    )
    ground_platforms = tuple(
        str(value).upper() for value in (ground_platforms or DEFAULT_GROUND_PLATFORMS)
    )
    aerial_mask = np.isin(cameras, aerial_platforms)
    ground_mask = np.isin(cameras, ground_platforms)

    protocols = {
        "A2G": (aerial_mask, ground_mask),
        "G2A": (ground_mask, aerial_mask),
        "G2G": (ground_mask, ground_mask),
        "A2A": (aerial_mask, aerial_mask),
    }

    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(
        f"AG-VPReID cameras: aerial={list(aerial_platforms)}, "
        f"ground={list(ground_platforms)}"
    )
    results = {}
    for name, (query_mask, gallery_mask) in protocols.items():
        scores = _evaluate_protocol(
            features,
            labels,
            cameras,
            query_mask,
            gallery_mask,
            metric,
        )
        msg_mgr.log_info(f"{name}: {scores}")
        for metric_name, value in scores.items():
            results[f"scalar/test_accuracy/{name}/{metric_name}"] = value
    return results


EVALUATOR_REGISTRY = {"evaluate_UAV_Ground": evaluate_UAV_Ground}

