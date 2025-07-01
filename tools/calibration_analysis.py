import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from calibration_sharpness import *

from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)

from sklearn.tree import DecisionTreeRegressor  # Needed for AdaBoost and Bagging

class MultiModelRecalibrationVisualizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.models = {
            "isotonic": [IsotonicRegression(out_of_bounds="clip") for _ in range(num_dims)],
            #"ridge": [Ridge() for _ in range(num_dims)],

            # Random Forest (100 trees)
            #"rf100": [RandomForestRegressor(n_estimators=50, random_state=42) for _ in range(num_dims)],

            # Gradient Boosting
            #"gb": [GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42) for _ in range(num_dims)],

            # Histogram Gradient Boosting (faster for larger datasets)
            #"hgb": [HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42) for _ in range(num_dims)],

            # AdaBoost with simple trees as base estimators
            # Uncomment and import DecisionTreeRegressor if you want to enable
            #"ada": [
            #     AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, random_state=42)
            #     for _ in range(num_dims)
            # ],

            # Bagging with decision trees
            # Uncomment and import DecisionTreeRegressor if you want to enable
            #"bagging": [
            #     BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=None), n_estimators=50, random_state=42)
            #     for _ in range(num_dims)
             #],

            # Gaussian Process
            #"gp": [
            #    GaussianProcessRegressor(
            #        kernel=RBF(length_scale=1.0),
            #        alpha=1e-3,  # noise level for numerical stability
            #        normalize_y=True,
            #        n_restarts_optimizer=0,
            #        random_state=42,
            #    )
            #    for _ in range(num_dims)
            #],
        }

    def _get_cdf(self, mu: np.ndarray, sigma: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return norm.cdf(gt, loc=mu, scale=sigma)

    def _fit_models(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, clip_percentile: float = 0.01):
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)

        for model_type in self.models:
            print(f"Fitting model type: {model_type}")
            for d in range(self.num_dims):
                print(d)
                x = cdf[:, d].reshape(-1)
                y = np.array([(x <= v).mean() for v in x])

                # Exclude extreme CDF values
                q_low, q_high = np.quantile(x, [clip_percentile, 1 - clip_percentile])
                mask = (x >= q_low) & (x <= q_high)

                x_clip = x[mask].reshape(-1, 1)
                y_clip = y[mask]

                model = self.models[model_type][d]
                model.fit(x_clip, y_clip)

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
        return np.mean(np.abs((observed_p - expected_p)))

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

    def quantify_improvement_Kfold(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, n_splits: int = 2):
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

    def run_with_holdout(
        self,
        mu_train: np.ndarray, var_train: np.ndarray, gt_train: np.ndarray,
        mu_test: np.ndarray, var_test: np.ndarray, gt_test: np.ndarray
    ):
        """Fit on train set, visualize calibration on test set."""
        self._fit_models(mu_train, var_train, gt_train)
        cdf_uncal = self._get_cdf(mu_test, np.sqrt(var_test), gt_test)

        cdf_calibrated = {
            model_type: self._apply_model(mu_test, var_test, gt_test, model_type)
            for model_type in self.models
        }

        self._plot_calibration_coverage(
            cdf_uncal=cdf_uncal,
            cdf_cal_dict=cdf_calibrated,
            model_labels=["Uncalibrated"] + [f"Recalibrated ({m})" for m in self.models.keys()],
        )

    def _estimate_variance_from_isotonic(self,
        model, y_support: np.ndarray
    ) -> float:
        """
        Estimate the variance of a random variable whose CDF is defined
        by a fitted isotonic regression model.

        Args:
            model: sklearn IsotonicRegression model (monotonic CDF estimator)
            y_support: 1D array of evenly spaced y values over the support of the variable

        Returns:
            Estimated variance (scalar)
        """
        # Get CDF values across support
        cdf_vals = model.predict(y_support)

        # Compute PDF as numerical derivative (finite difference)
        pdf_vals = np.diff(cdf_vals) / np.diff(y_support)
        y_midpoints = 0.5 * (y_support[:-1] + y_support[1:])  # match PDF shape

        # Normalize in case model is not strictly [0,1]
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        pdf_vals /= np.sum(pdf_vals)  # ensure integrates to 1

        # Compute mean
        mean = np.sum(pdf_vals * y_midpoints)

        # Compute variance
        var = np.sum(pdf_vals * (y_midpoints - mean) ** 2)

        return var

    
    def quantify_improvement_with_holdout(
        self,
        mu_train: np.ndarray, var_train: np.ndarray, gt_train: np.ndarray,
        mu_test: np.ndarray, var_test: np.ndarray, gt_test: np.ndarray,
    ):
        """Fit on train set, report calibration error and per-dimension sharpness on test set."""
        self._fit_models(mu_train, var_train, gt_train)

        # --- Uncalibrated ---
        cdf_uncal = self._get_cdf(mu_test, np.sqrt(var_test), gt_test)
        unc_error = self._compute_calibration_error(cdf_uncal)
        unc_sharpness_vec = np.mean(var_test)  # per-dim average σ²
        print(f"Calibration Error (Uncalibrated): {unc_error:.6f}")
        print(f"Sharpness (Uncalibrated): {unc_sharpness_vec}")

        # --- Calibrated ---
        for model_type in self.models:
            print("################################# ", model_type, " ###########################################")
            cdf_cal = self._apply_model(mu_test, var_test, gt_test, model_type)
            err = self._compute_calibration_error(cdf_cal)
            print(f"Calibration Error (Recalibrated - {model_type}): {err:.6f}")

            # Compute sharpness as variance of predictive distribution (only if monotonic model)
            if model_type in ["isotonic"]:  # list of monotonic models
                sharpness_vals = []
                for d in range(self.num_dims):
                    model = self.models[model_type][d]
                    # Use dense y support for numerical integration (per dimension)
                    #y_support = np.linspace(np.min(gt_test[:, d]), np.max(gt_test[:, d]), 500)
                    q_low, q_high = np.quantile(gt_test[:, d], [0.01, 0.99])
                    y_support = np.linspace(q_low, q_high, 500)
                    var_d = self._estimate_variance_from_isotonic(model, y_support)
                    sharpness_vals.append(var_d)
                sharpness_vec = np.sqrt(np.stack(sharpness_vals, axis=0))
                print(f"Sharpness Vector (Recalibrated - {model_type}):")
                print(sharpness_vec)
                print("mean ", np.mean(sharpness_vec))
            else:
                print(f"(Skipping sharpness: {model_type} is not a monotonic CDF model)")


            




def main():
    mu_train = np.load("prediction_data/all_predictions_validation.npy")
    var_train = np.load("prediction_data/all_sigmas_validation.npy")
    gt_train = np.load("prediction_data/all_ground_truths_validation.npy")

    mu_test = np.load("prediction_data/all_predictions_test.npy")
    var_test = np.load("prediction_data/all_sigmas_test.npy")
    gt_test = np.load("prediction_data/all_ground_truths_test.npy")

    
    # Run recalibration and plot
    #visualizer = IsotonicRecalibrationVisualizer(num_dims=78)
    visualizer = MultiModelRecalibrationVisualizer(num_dims = 78)
    #visualizer.run(mu, var, gt)
    visualizer.quantify_improvement_with_holdout(mu_train, var_train, gt_train, 
                                                 mu_test, var_test, gt_test)
    

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
