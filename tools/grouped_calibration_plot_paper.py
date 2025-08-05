import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from typing import List, Dict, Tuple, Any
import os
import matplotlib.cm as cm
from matplotlib.colors import to_hex



class GroupedModelComparisonVisualizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.models = {}  # Maps model label -> list of fitted IsotonicRegression (per dim)

    def _get_cdf(self, mu: np.ndarray, sigma: np.ndarray, gt: np.ndarray, laplace: bool = False) -> np.ndarray:
        if laplace:
            b = np.sqrt(sigma**2 / 2)
            return np.where(gt < mu,
                            0.5 * np.exp((gt - mu) / b),
                            1 - 0.5 * np.exp(-(gt - mu) / b))
        else:
            return norm.cdf(gt, loc=mu, scale=sigma)

    def _get_cdf_from_samples(self, samples: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Compute empirical CDF at ground truth from predictive samples (shape: [N, D, S])"""
        return (samples <= gt[:, :, None]).mean(axis=2)

    def _compute_empirical_cdf(self, cdf: np.ndarray, num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        expected_p = np.linspace(0.01, 0.99, num_bins)
        flat = cdf.flatten()
        observed_p = np.array([(flat <= p).mean() for p in expected_p])
        return expected_p, observed_p

    def fit_calibrators(self, model_data: List[Dict]) -> None:
        """Train and store calibrators per model (by label) using validation data."""
        for entry in model_data:
            mu, var, gt = entry['mu'], entry['var'], entry['gt']
            label = entry['label']
            likelihood = entry['likelihood']

            # Handle sample-based models (e.g., Normalizing Flow)
            if 'cdf' in entry and entry['cdf'].ndim == 3:
                cdf = self._get_cdf_from_samples(entry['cdf'], gt)
            else:
                sigma = np.sqrt(var)
                cdf = self._get_cdf(mu, sigma, gt, laplace=(likelihood == 'laplace'))

            calibrators = []
            for d in range(self.num_dims):
                x = cdf[:, d]
                y = np.array([(x <= v).mean() for v in x])
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(x, y)
                calibrators.append(model)

            self.models[label] = calibrators

    def plot_grouped_cdfs(
        self,
        model_data: List[Dict],
        save_path: str = None,
        font_scale: float = 3.0,
        num_bins: int = 100,
    ):
        base_fontsize = 10 * font_scale
        plt.rcParams.update({
            "font.family": "serif",
            "axes.titlesize": base_fontsize * 1.2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.9,
        })

        fig, ax = plt.subplots(figsize=(18, 18))

        model_keys = list({(m['latent'], m['likelihood']) for m in model_data})
        cmap = cm.get_cmap("tab10", len(model_keys))
        color_map = {key: to_hex(cmap(i)) for i, key in enumerate(model_keys)}
        linestyle_map = {False: "dotted", True: "solid"}

        for entry in model_data:
            mu, var, gt = entry['mu'], entry['var'], entry['gt']
            label = entry['label']
            latent = entry['latent']
            likelihood = entry['likelihood']
            model_key = (latent, likelihood)
            color = color_map[model_key]

            # Uncalibrated CDF
            if 'cdf' in entry and entry['cdf'].ndim == 3:
                cdf_uncal = self._get_cdf_from_samples(entry['cdf'], gt)
            else:
                sigma = np.sqrt(var)
                laplace_flag = (likelihood == 'laplace')
                cdf_uncal = self._get_cdf(mu, sigma, gt, laplace=laplace_flag)

            exp_p, obs_p = self._compute_empirical_cdf(cdf_uncal, num_bins)
            ax.plot(exp_p, obs_p, linestyle=linestyle_map[False], color=color, linewidth=2.5,
                    label=f"{label} (Uncal)")

            # Calibrated CDF
            if label not in self.models:
                raise ValueError(f"Missing calibrator for model '{label}'")
            cdf_cal = np.zeros_like(cdf_uncal)
            for d in range(self.num_dims):
                model = self.models[label][d]
                cdf_cal[:, d] = model.predict(cdf_uncal[:, d])
            exp_p, obs_p = self._compute_empirical_cdf(cdf_cal, num_bins)
            ax.plot(exp_p, obs_p, linestyle=linestyle_map[True], color=color, linewidth=2.5,
                    label=f"{label} (Cal)")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
        ax.set_xlabel("Expected Confidence Level")
        ax.set_ylabel("Observed Confidence Level")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":")

        handles, labels = ax.get_legend_handles_labels()
        def sort_key(label):
            if "Cal" in label:
                return (0, label)
            elif "Uncal" in label:
                return (1, label)
            elif "Perfect" in label:
                return (2, label)
            else:
                return (3, label)

        sorted_items = sorted(zip(handles, labels), key=lambda x: sort_key(x[1]))
        sorted_handles, sorted_labels = zip(*sorted_items)
        ax.legend(sorted_handles, sorted_labels, frameon=False)

        if save_path:
            plt.savefig(save_path, dpi=500)
        plt.show()

    def plot_grouped_cdfs_2(
        self,
        model_data: List[Dict],
        save_path: str = None,
        font_scale: float = 3.0,
        num_bins: int = 100,
    ):
        """
        Plot uncalibrated and calibrated calibration curves side by side.
        """

        def format_label(latent: str, likelihood: str, label_fallback: str) -> str:
            # Always keep Evidential and NF names
            if label_fallback in ["Evidential Regression", "Normalizing Flow"]:
                return label_fallback

            # Base latent–likelihood combos without suffixes
            base_combos = {
                "gaussian-gaussian",
                "gaussian-laplace",
                "laplace-gaussian",
                "laplace-laplace",
            }
            # Keep the label if it contains extra info beyond base combos
            if label_fallback.lower().replace("–", "-") not in base_combos:
                return label_fallback

            # Otherwise format cleanly
            def fmt(x):
                return "Gauss." if x == "gaussian" else "Laplace"
            return f"RPP {fmt(latent.lower())} {fmt(likelihood.lower())}"

        base_fontsize = 10 * font_scale
        plt.rcParams.update({
            "font.family": "serif",
            "axes.titlesize": base_fontsize * 1.2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.9,
        })

        fig, (ax_uncal, ax_cal) = plt.subplots(1, 2, figsize=(36, 18))

        # Collect formatted labels
        model_keys = list({
            format_label(m['latent'], m['likelihood'], m['label']) for m in model_data
        })
        cmap = cm.get_cmap("tab10", len(model_keys))
        color_map = {key: to_hex(cmap(i)) for i, key in enumerate(model_keys)}

        for entry in model_data:
            label_raw = entry['label']
            label = format_label(entry['latent'], entry['likelihood'], label_raw)
            mu, var, gt = entry['mu'], entry['var'], entry['gt']
            color = color_map[label]

            # Get CDF
            if 'cdf' in entry and entry['cdf'].ndim == 3:
                cdf_uncal = self._get_cdf_from_samples(entry['cdf'], gt)
            else:
                sigma = np.sqrt(var)
                laplace_flag = (entry['likelihood'] == 'laplace')
                cdf_uncal = self._get_cdf(mu, sigma, gt, laplace=laplace_flag)

            # Uncalibrated plot
            exp_p, obs_p = self._compute_empirical_cdf(cdf_uncal, num_bins)
            ax_uncal.plot(exp_p, obs_p, linestyle="-", color=color, linewidth=2.5, label=label)

            # Calibrated plot
            if label_raw not in self.models:
                raise ValueError(f"Missing calibrator for model '{label_raw}'")
            cdf_cal = np.zeros_like(cdf_uncal)
            for d in range(self.num_dims):
                model = self.models[label_raw][d]
                cdf_cal[:, d] = model.predict(cdf_uncal[:, d])
            exp_p, obs_p = self._compute_empirical_cdf(cdf_cal, num_bins)
            ax_cal.plot(exp_p, obs_p, linestyle="-", color=color, linewidth=2.5, label=label)

        for ax, title in zip([ax_uncal, ax_cal], ["a)", "b)"]):
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
            ax.set_xlabel("Expected Confidence Level")
            ax.set_ylabel("Observed Confidence Level")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, linestyle=":")
            ax.set_title(title, loc="left", fontsize=base_fontsize * 1.4)

        # Legend only on uncalibrated plot
        handles, labels = ax_uncal.get_legend_handles_labels()
        ax_uncal.legend(handles, labels, frameon=False)

        if save_path:
            plt.savefig(save_path, dpi=500)
        plt.show()



def load_model_data(
    mu_path: str,
    var_path: str,
    gt_path: str,
    label: str,
    latent: str,
    likelihood: str,
    **kwargs: str,
) -> Dict[str, Any]:
    """
    Load predictive model outputs and attach metadata for grouped calibration plots.

    Args:
        mu_path: Path to .npy file with predicted means (shape: [N, D])
        var_path: Path to .npy file with predicted variances (shape: [N, D])
        gt_path: Path to .npy file with ground truth targets (shape: [N, D])
        label: Label used in the calibration plot legend
        latent: Type of latent distribution used in the model ('gaussian' or 'laplace')
        likelihood: Type of likelihood distribution ('gaussian' or 'laplace')
        **kwargs: Additional named npy file paths (e.g., cdf_path='cdf_val_nf.npy')

    Returns:
        Dictionary with model predictions and descriptive metadata.
    """
    data = {
        "mu": np.load(mu_path),
        "var": np.load(var_path),
        "gt": np.load(gt_path),
        "label": label,
        "latent": latent.lower(),
        "likelihood": likelihood.lower(),
    }

    # Load any additional arrays (like cdf, scores, etc.)
    for key, path in kwargs.items():
        data[key.replace("_path", "")] = np.load(path)

    return data


if __name__ == "__main__":
    os.chdir('/home/jonas/code/RadProPoser/tools/calibration_analysis')

    # Replace this with the true output dimension of your dataset
    NUM_DIMS = 78

    val_model_data = [
        load_model_data("mu_val_gauss_gauss_cov.npy",
                        "var_val_gauss_gauss_cov.npy",
                        "gt_val_gauss_gauss_cov.npy",
                        label="Gauss–Gauss-Cov", latent="gaussian", likelihood="gaussian"),
        load_model_data("mu_val_gauss_gauss.npy",
                        "var_val_gauss_gauss.npy",
                        "gt_val_gauss_gauss.npy",
                        label="Gauss–Gauss", latent="gaussian", likelihood="gaussian"),

        load_model_data("mu_val_gauss_laplace.npy",
                        "var_val_gauss_laplace.npy",
                        "gt_val_gauss_laplace.npy",
                        label="Gauss–Laplace", latent="gaussian", likelihood="laplace"),

        load_model_data("mu_val_laplace_gauss.npy",
                        "var_val_laplace_gauss.npy",
                        "gt_val_laplace_gauss.npy",
                        label="Laplace–Gauss", latent="laplace", likelihood="gaussian"),

        load_model_data("mu_val_laplace_laplace.npy",
                        "var_val_laplace_laplace.npy",
                        "gt_val_laplace_laplace.npy",
                        label="Laplace–Laplace", latent="laplace", likelihood="laplace"),

        load_model_data("mu_val_nf.npy",
                "var_val_nf.npy",
                "gt_val_nf.npy",
                label="Normalizing Flow", latent="nf", likelihood="nf",
                cdf_path="cdf_val_nf.npy"),
        
        load_model_data("mu_val_evd.npy",
                        "var_val_evd.npy",
                        "gt_val_evd.npy",
                        label="Evidential Regression", latent="gaussian", likelihood="gaussian"),
    ]

    test_model_data = [
        load_model_data("mu_testing_gauss_gauss_cov.npy",
                        "var_testing_gauss_gauss_cov.npy",
                        "gt_testing_gauss_gauss_cov.npy",
                        label="Gauss–Gauss-Cov", latent="gaussian", likelihood="gaussian"),

        load_model_data("mu_testing_gauss_gauss.npy",
                        "var_testing_gauss_gauss.npy",
                        "gt_testing_gauss_gauss.npy",
                        label="Gauss–Gauss", latent="gaussian", likelihood="gaussian"),

        load_model_data("mu_testing_gauss_laplace.npy",
                        "var_testing_gauss_laplace.npy",
                        "gt_testing_gauss_laplace.npy",
                        label="Gauss–Laplace", latent="gaussian", likelihood="laplace"),

        load_model_data("mu_testing_laplace_gauss.npy",
                        "var_testing_laplace_gauss.npy",
                        "gt_testing_laplace_gauss.npy",
                        label="Laplace–Gauss", latent="laplace", likelihood="gaussian"),

        load_model_data("mu_testing_laplace_laplace.npy",
                        "var_testing_laplace_laplace.npy",
                        "gt_testing_laplace_laplace.npy",
                        label="Laplace–Laplace", latent="laplace", likelihood="laplace"),

        load_model_data("mu_testing_nf.npy",
                "var_testing_nf.npy",
                "gt_testing_nf.npy",
                label="Normalizing Flow", latent="nf", likelihood="nf",
                cdf_path="cdf_testing_nf.npy"),

        load_model_data("mu_testing_evd.npy",
                        "var_testing_evd.npy",
                        "gt_testing_evd.npy",
                        label="Evidential Regression", latent="gaussian", likelihood="gaussian"),
    ]

    visualizer = GroupedModelComparisonVisualizer(num_dims=NUM_DIMS)
    visualizer.fit_calibrators(val_model_data)
    visualizer.plot_grouped_cdfs_2(test_model_data, save_path="calibration_comparison.pdf")
