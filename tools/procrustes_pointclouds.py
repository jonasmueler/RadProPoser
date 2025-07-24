import os
import re
import numpy as np
import torch
from typing import Tuple


def compute_p_mpjpe(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.view(-1, 26, 3)
    targets = targets.view(-1, 26, 3)

    aligned_preds = torch.stack([
        procrustes_torch(tgt, pred) for pred, tgt in zip(preds, targets)
    ])

    diff = aligned_preds - targets
    dist = torch.norm(diff, dim=-1)
    return dist.mean().item()


def procrustes_torch(X: torch.Tensor, Y: torch.Tensor, scaling: bool = True, reflection: str = 'best') -> torch.Tensor:
    muX = X.mean(dim=0, keepdim=True)
    muY = Y.mean(dim=0, keepdim=True)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2).sum()
    ssY = (Y0 ** 2).sum()

    normX = torch.sqrt(ssX)
    normY = torch.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    A = X0.T @ Y0
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = V @ U.T

    if reflection != 'best':
        have_reflection = torch.det(T) < 0
        if reflection != have_reflection:
            V[:, -1] *= -1
            S[-1] *= -1
            T = V @ U.T

    traceTA = S.sum()

    if scaling:
        Z = normX * traceTA * (Y0 @ T) + muX
    else:
        Z = normY * (Y0 @ T) + muX

    return Z


def evaluate_all(directory: str):
    pattern = r"_(an\d+_p\d+)\.npy"

    files = os.listdir(directory)
    gt_map = {}
    pred_map = {}

    for file in files:
        match = re.search(pattern, file)
        if match:
            key = match.group(1)
            full_path = os.path.join(directory, file)
            if file.startswith("ground_truth"):
                gt_map[key] = full_path
            elif file.startswith("predicted"):
                pred_map[key] = full_path

    # Intersect and sort by condition key
    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))

    for key in common_keys:
        gt_path = gt_map[key]
        pred_path = pred_map[key]

        gt = torch.from_numpy(np.load(gt_path)).float()
        pred = torch.from_numpy(np.load(pred_path)).float()

        mpjpe = compute_p_mpjpe(pred, gt)
        print(f"avg_mpjpe_{key}\t{mpjpe}")


# Example usage:
evaluate_all("/home/jonas/code/RadProPoser/tools/prediction_data/metrics/metrics1")
