import torch 
import numpy as np
from calibration_sharpness import *

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from typing import List, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold



class IsotonicRecalibrationVisualizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.models = [IsotonicRegression(out_of_bounds="clip") for _ in range(num_dims)]

    def _get_cdf(self, mu: np.ndarray, sigma: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Compute predicted CDF values for each sample and dimension."""
        return norm.cdf(gt, loc=mu, scale=sigma)

    def _fit_recalibration_models(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray) -> None:
        """Fit isotonic regression models per dimension."""
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)  # shape: (N, D)
        for d in range(self.num_dims):
            self.models[d].fit(cdf[:, d], np.sort(cdf[:, d]))

    def _apply_recalibration(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Apply fitted isotonic models to calibrate CDFs."""
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)
        recalibrated = np.zeros_like(cdf)
        for d in range(self.num_dims):
            recalibrated[:, d] = self.models[d].predict(cdf[:, d])
        return recalibrated

    def _plot_calibration_coverage(
        self,
        cdf_uncal: np.ndarray,
        cdf_cal: np.ndarray,
        model_labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        num_bins: int = 50,
        fill_alpha: float = 0.25,
        font_scale: float = 4,
    ):
        expected_p = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
        base_fontsize = 10 * font_scale
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": base_fontsize * 1.2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.9,
        })

        fig, ax = plt.subplots(figsize=(6, 6))

        for i, (cdf_vals, label, color) in enumerate(zip(
            [cdf_uncal, cdf_cal],
            model_labels or ["Uncalibrated", "Recalibrated"],
            colors or [None, None],
        )):
            cdf_vals_flat = cdf_vals.flatten()
            observed_p = [(cdf_vals_flat <= q).mean() for q in expected_p]

            ax.plot(expected_p, observed_p, label=label, linewidth=2, color=color)
            #ax.fill_between(
            #    expected_p,
            #    np.maximum(0, np.array(observed_p) - 0.02),
            #    np.minimum(1, np.array(observed_p) + 0.02),
            #    alpha=fill_alpha,
            #    color=color,
            #)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
        ax.set_xlabel("Expected Confidence Level")
        ax.set_ylabel("Observed Confidence Level")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":")
        ax.legend(frameon=False)
        plt.show()
        
    def _compute_calibration_errors(self, 
                                    cdf_uncal: np.ndarray, 
                                    cdf_cal: np.ndarray, 
                                    num_bins: int = 100) -> tuple[float, float]:
        """
        Compute the calibration error before and after isotonic recalibration.

        Args:
            cdf_uncal (np.ndarray): Uncalibrated CDF values, shape (N, D).
            cdf_cal (np.ndarray): Recalibrated CDF values, shape (N, D).
            num_bins (int): Number of probability thresholds to evaluate.

        Returns:
            Tuple[float, float]: (uncalibrated_error, recalibrated_error)
        """
        expected_p = np.linspace(0.01, 0.99, num_bins)

        def error(cdf: np.ndarray):
            flat = cdf.flatten()
            observed_p = np.array([(flat <= p).mean() for p in expected_p])
            return np.mean((observed_p - expected_p) ** 2)

        return error(cdf_uncal), error(cdf_cal)

    def run(
        self,
        mu: np.ndarray,
        var: np.ndarray,
        gt: np.ndarray,
    ):
        """
        Train recalibration models and plot calibration curves.

        Args:
            mu (np.ndarray): Shape (N, D), predicted means.
            var (np.ndarray): Shape (N, D), predicted variances.
            gt (np.ndarray): Shape (N, D), ground truth values.
        """
        cdf_uncal = self._get_cdf(mu, np.sqrt(var), gt)
        self._fit_recalibration_models(mu, var, gt)
        cdf_cal = self._apply_recalibration(mu, var, gt)
        self._plot_calibration_coverage(
            cdf_uncal=cdf_uncal,
            cdf_cal=cdf_cal,
            model_labels=["Uncalibrated", "Recalibrated"],
        )
        
    def quantify_improvement(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        """
        Print and return calibration error before and after isotonic recalibration.

        Args:
            mu (np.ndarray): Predicted means, shape (N, D).
            var (np.ndarray): Predicted variances, shape (N, D).
            gt (np.ndarray): Ground truth values, shape (N, D).

        Returns:
            Tuple[float, float]: (uncalibrated_error, recalibrated_error)
        """
        cdf_uncal = self._get_cdf(mu, np.sqrt(var), gt)
        self._fit_recalibration_models(mu, var, gt)
        cdf_cal = self._apply_recalibration(mu, var, gt)

        err_uncal, err_cal = self._compute_calibration_errors(cdf_uncal, cdf_cal)

        print(f"Calibration Error (Before): {err_uncal:.6f}")
        print(f"Calibration Error (After):  {err_cal:.6f}")

        return err_uncal, err_cal


import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from typing import List, Optional

class MultiModelRecalibrationVisualizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.models = {
            "isotonic": [IsotonicRegression(out_of_bounds="clip") for _ in range(num_dims)],
            "ridge": [Ridge() for _ in range(num_dims)],
            #"lasso": [Lasso() for _ in range(num_dims)],
            #"elasticnet": [ElasticNet() for _ in range(num_dims)],
            "rf2": [RandomForestRegressor(n_estimators=2) for _ in range(num_dims)],
            "rf5": [RandomForestRegressor(n_estimators=5) for _ in range(num_dims)],
            "rf10": [RandomForestRegressor(n_estimators=10) for _ in range(num_dims)],
            "rf20": [RandomForestRegressor(n_estimators=20) for _ in range(num_dims)],
            #"gp": [GaussianProcessRegressor(optimizer="fmin_l_bfgs_b",
            #    n_restarts_optimizer=1,        # Reduce restarts
            #    alpha=1e-6,                     # Stabilize numerics
            #    normalize_y=True) for _ in range(num_dims)]
                    }

    def _get_cdf(self, mu: np.ndarray, sigma: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return norm.cdf(gt, loc=mu, scale=sigma)

    def _fit_models(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray):
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)
        for model_type in self.models:
            print(model_type)
            for d in range(self.num_dims):
                print(d)
                self.models[model_type][d].fit(cdf[:, d].reshape(-1, 1), np.sort(cdf[:, d]))

    def _apply_model(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, model_type: str) -> np.ndarray:
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)
        recalibrated = np.zeros_like(cdf)
        for d in range(self.num_dims):
            recalibrated[:, d] = self.models[model_type][d].predict(cdf[:, d].reshape(-1, 1))
        return recalibrated

import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from typing import List

class MultiModelRecalibrationVisualizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.models = {
            #"isotonic": [IsotonicRegression(out_of_bounds="clip") for _ in range(num_dims)],
            #"ridge": [Ridge() for _ in range(num_dims)],
            #"rf_joint": RandomForestRegressor(n_estimators=5),
            #"rf2": [RandomForestRegressor(n_estimators=2) for _ in range(num_dims)],
            #"rf5": [RandomForestRegressor(n_estimators=5) for _ in range(num_dims)],
            #"rf10": [RandomForestRegressor(n_estimators=10) for _ in range(num_dims)],
            "rf80": [RandomForestRegressor(n_estimators=80) for _ in range(num_dims)],
        }

    def _get_cdf(self, mu: np.ndarray, sigma: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return norm.cdf(gt, loc=mu, scale=sigma)

    def _fit_models(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray):
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)
        for model_type in self.models:
            print(model_type)
            if model_type == "rf_joint":
                self.models[model_type].fit(cdf, np.sort(cdf, axis=0))
            else:
                for d in range(self.num_dims):
                    print(d)
                    self.models[model_type][d].fit(cdf[:, d].reshape(-1, 1), np.sort(cdf[:, d]))

    def _apply_model(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, model_type: str) -> np.ndarray:
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)
        if model_type == "rf_joint":
            return self.models[model_type].predict(cdf).reshape(cdf.shape)
        else:
            recalibrated = np.zeros_like(cdf)
            for d in range(self.num_dims):
                recalibrated[:, d] = self.models[model_type][d].predict(cdf[:, d].reshape(-1, 1))
            return recalibrated

    def _plot_calibration_coverage(
        self,
        cdf_uncal: np.ndarray,
        cdf_cal_dict: dict,
        model_labels: List[str],
        num_bins: int = 50,
        font_scale: float = 4,
    ):
        expected_p = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
        base_fontsize = 10 * font_scale
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": base_fontsize * 1.2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.9,
        })

        fig, ax = plt.subplots(figsize=(6, 6))

        def plot_cdf(cdf_vals: np.ndarray, label: str):
            flat = cdf_vals.flatten()
            observed_p = [(flat <= q).mean() for q in expected_p]
            ax.plot(expected_p, observed_p, label=label, linewidth=2)

        plot_cdf(cdf_uncal, "Uncalibrated")
        for method, cdf_vals in cdf_cal_dict.items():
            plot_cdf(cdf_vals, f"Recalibrated ({method})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
        ax.set_xlabel("Expected Confidence Level")
        ax.set_ylabel("Observed Confidence Level")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":")
        ax.legend(frameon=False)
        plt.show()

    def _compute_calibration_error(self, cdf: np.ndarray, num_bins: int = 100) -> float:
        expected_p = np.linspace(0.01, 0.99, num_bins)
        flat = cdf.flatten()
        observed_p = np.array([(flat <= p).mean() for p in expected_p])
        return np.mean((observed_p - expected_p) ** 2)

    def run(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray):
        cdf_uncal = self._get_cdf(mu, np.sqrt(var), gt)
        self._fit_models(mu, var, gt)

        cdf_calibrated = {
            model_type: self._apply_model(mu, var, gt, model_type)
            for model_type in self.models
        }

        self._plot_calibration_coverage(
            cdf_uncal=cdf_uncal,
            cdf_cal_dict=cdf_calibrated,
            model_labels=["Uncalibrated"] + [f"Recalibrated ({m})" for m in self.models.keys()],
        )

    def quantify_improvement(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray):
        cdf_uncal = self._get_cdf(mu, np.sqrt(var), gt)
        self._fit_models(mu, var, gt)

        print(f"Calibration Error (Uncalibrated): {self._compute_calibration_error(cdf_uncal):.6f}")
        for model_type in self.models:
            cdf_cal = self._apply_model(mu, var, gt, model_type)
            err = self._compute_calibration_error(cdf_cal)
            print(f"Calibration Error (Recalibrated - {model_type}): {err:.6f}")

    def quantify_improvement_Kfold(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, n_splits: int = 5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        unc_errors = []
        cal_errors = {model_type: [] for model_type in self.models}

        for train_idx, val_idx in kf.split(mu):
            mu_train, var_train, gt_train = mu[train_idx], var[train_idx], gt[train_idx]
            mu_val, var_val, gt_val = mu[val_idx], var[val_idx], gt[val_idx]

            self._fit_models(mu_train, var_train, gt_train)

            cdf_uncal = self._get_cdf(mu_val, np.sqrt(var_val), gt_val)
            unc_errors.append(self._compute_calibration_error(cdf_uncal))

            for model_type in self.models:
                cdf_cal = self._apply_model(mu_val, var_val, gt_val, model_type)
                cal_errors[model_type].append(self._compute_calibration_error(cdf_cal))

        print(f"Cross-Validated Calibration Error (Uncalibrated): {np.mean(unc_errors):.6f} ± {np.std(unc_errors):.6f}")
        for model_type in self.models:
            mean_err = np.mean(cal_errors[model_type])
            std_err = np.std(cal_errors[model_type])
            print(f"Cross-Validated Calibration Error (Recalibrated - {model_type}): {mean_err:.6f} ± {std_err:.6f}")



def main():
    mu = np.load("prediction_data/all_predictions.npy")
    var = np.load("prediction_data/all_sigmas.npy")
    gt = np.load("prediction_data/all_ground_truths.npy")
    
    # Run recalibration and plot
    #visualizer = IsotonicRecalibrationVisualizer(num_dims=78)
    visualizer = MultiModelRecalibrationVisualizer(num_dims = 78)
    #visualizer.run(mu, var, gt)
    visualizer.quantify_improvement_Kfold(mu, var, gt)
    

def main_test():
    preds = [torch.from_numpy(np.load("prediction_data/all_predictions.npy")).unsqueeze(dim = 0)]
    vars = [torch.from_numpy(np.load("prediction_data/all_sigmas.npy")).unsqueeze(dim = 0)]
    gts = torch.from_numpy(np.load("prediction_data/all_ground_truths.npy")).unsqueeze(dim = 0)
    
    plot_calibration_coverage(preds, vars, gts)
    
    cal_error, sharpness, (expected, observed) = compute_calibration_and_sharpness(torch.from_numpy(np.load("prediction_data/all_predictions.npy")), 
                                      torch.from_numpy(np.load("prediction_data/all_sigmas.npy")), 
                                      torch.from_numpy(np.load("prediction_data/all_ground_truths.npy")))
    
    print(cal_error)
    print(sharpness)
    #print(expected)
    #print(observed)
    
    
if __name__ == "__main__":
    main()
