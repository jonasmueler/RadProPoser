import os
import numpy as np
import torch
from typing import Tuple, Dict
from scipy.stats import pearsonr


class CalibrationDataLoader:
    """Loads and processes prediction, variance, and ground truth data."""

    def __init__(self, model_name: str, root_dir: str):
        """
        Args:
            model_name (str): Model identifier (e.g., 'gauss_gauss').
            root_dir (str): Root path to the calibration_analysis folder.
        """
        self.model_name = model_name
        self.root_dir = root_dir

        # Load and concatenate val + test data
        self.mu = self._load_and_concat("mu")
        self.var = self._load_and_concat("var")
        self.gt = self._load_and_concat("gt")

        print(f"[INFO] Loaded model: {model_name}")
        print(f"  ➤ Predictions: {self.mu.shape}")
        print(f"  ➤ Variances:   {self.var.shape}")
        print(f"  ➤ Ground truth: {self.gt.shape}")

    def _load_and_concat(self, prefix: str) -> np.ndarray:
        val_path = os.path.join(self.root_dir, f"{prefix}_val_{self.model_name}.npy")
        test_path = os.path.join(self.root_dir, f"{prefix}_testing_{self.model_name}.npy")

        val_data = np.load(val_path)
        test_data = np.load(test_path)
        return np.concatenate([val_data, test_data], axis=0)

    @staticmethod
    def compute_mpjpe(preds: torch.Tensor, targets: torch.Tensor, keypoints: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean per joint position error (MPJPE).

        Args:
            preds (torch.Tensor): Predicted joint positions, shape (N, 78).
            targets (torch.Tensor): Ground truth joint positions, shape (N, 78).
            keypoints (bool): If True, returns per-keypoint mean/std.

        Returns:
            Tuple of MPJPE mean and std.
        """
        preds = preds.view(preds.size(0), 26, 3)
        targets = targets.view(targets.size(0), 26, 3)

        diff = preds - targets
        dist = torch.norm(diff, dim=-1)  # (N, 26)

        if keypoints:
            return dist.mean(dim=0), dist.std(dim=0)
        return dist.mean(), dist.std()

    def compute_error_and_variance_correlations(self) -> Dict[str, np.ndarray]:
        """Computes Pearson correlation between error and predicted variance.

        Returns:
            Dictionary with:
              - per_keypoint_corr: (26,) array of correlations
              - overall_corr: float
        """
        preds = torch.tensor(self.mu)
        gts = torch.tensor(self.gt)
        vars_np = self.var  # (N, 78)

        # Compute per-keypoint errors
        errors = torch.norm((preds - gts).view(-1, 26, 3), dim=-1).numpy()  # (N, 26)

        # Convert variance to per-keypoint (N, 26)
        var_per_keypoint = np.sum(vars_np.reshape(-1, 26, 3), axis=2)  # (N, 26)

        # Compute Pearson correlation per keypoint
        per_keypoint_corr = np.zeros(26)
        for k in range(26):
            per_keypoint_corr[k], _ = pearsonr(errors[:, k], var_per_keypoint[:, k])

        # Flatten both for overall correlation
        overall_corr, _ = pearsonr(errors.flatten(), var_per_keypoint.flatten())

        return {
            "per_keypoint_corr": per_keypoint_corr,
            "overall_corr": overall_corr
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gauss_gauss)")
    parser.add_argument("--root_dir", type=str, default=".", help="Path to calibration_analysis directory")
    args = parser.parse_args()

    loader = CalibrationDataLoader(args.model, args.root_dir)

    # Compute MPJPE
    preds_torch = torch.tensor(loader.mu)
    gts_torch = torch.tensor(loader.gt)
    mpjpe_mean, mpjpe_std = CalibrationDataLoader.compute_mpjpe(preds_torch, gts_torch)

    print(f"\nMPJPE: {mpjpe_mean:.2f} ± {mpjpe_std:.2f} mm")

    # Compute correlations
    correlations = loader.compute_error_and_variance_correlations()
    print("\nPer-keypoint Pearson correlation (error vs variance):")
    for i, c in enumerate(correlations["per_keypoint_corr"]):
        print(f"  Keypoint {i:2d}: r = {c:.3f}")

    print(f"\nOverall Pearson correlation: r = {correlations['overall_corr']:.3f}")
