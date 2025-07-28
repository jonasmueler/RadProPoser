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
        """
        Plot both calibrated and uncalibrated calibration curves per model.
        model_data should contain keys: 'mu', 'var', 'gt', 'label', 'latent', 'likelihood'
        """

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
            sigma = np.sqrt(var)
            laplace_flag = (likelihood == 'laplace')
            model_key = (latent, likelihood)
            color = color_map[model_key]

            # Uncalibrated CDF
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
        #ax.legend(frameon=False)

        # reorder legend parameters
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


def load_model_data(
    mu_path: str,
    var_path: str,
    gt_path: str,
    label: str,
    latent: str,
    likelihood: str,
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

    Returns:
        Dictionary with model predictions and descriptive metadata.
    """
    mu = np.load(mu_path)
    var = np.load(var_path)
    gt = np.load(gt_path)

    return {
        "mu": mu,
        "var": var,
        "gt": gt,
        "label": label,
        "latent": latent.lower(),
        "likelihood": likelihood.lower(),
    }


if __name__ == "__main__":
    os.chdir('/home/jonas/code/RadProPoser/tools/prediction_data')

    # Replace this with the true output dimension of your dataset
    NUM_DIMS = 78

    val_model_data = [
        load_model_data("all_predictions_validation_gaussian_gaussian.npy",
                        "all_sigmas_validation_gaussian_gaussian.npy",
                        "all_ground_truths_validation_gaussian_gaussian.npy",
                        label="Gauss–Gauss", latent="gaussian", likelihood="gaussian"),

        load_model_data("all_predictions_validation_gaussian_laplace.npy",
                        "all_sigmas_validation_gaussian_laplace.npy",
                        "all_ground_truths_validation_gaussian_laplace.npy",
                        label="Gauss–Laplace", latent="gaussian", likelihood="laplace"),

        load_model_data("all_predictions_validation_laplace_gaussian.npy",
                        "all_sigmas_validation_laplace_gaussian.npy",
                        "all_ground_truths_validation_laplace_gaussian.npy",
                        label="Laplace–Gauss", latent="laplace", likelihood="gaussian"),

        load_model_data("all_predictions_validation_laplace_laplace.npy",
                        "all_sigmas_validation_laplace_laplace.npy",
                        "all_ground_truths_validation_laplace_laplace.npy",
                        label="Laplace–Laplace", latent="laplace", likelihood="laplace"),
    ]

    test_model_data = [
        load_model_data("all_predictions_testing_gaussian_gaussian.npy",
                        "all_sigmas_testing_gaussian_gaussian.npy",
                        "all_ground_truths_testing_gaussian_gaussian.npy",
                        label="Gauss–Gauss", latent="gaussian", likelihood="gaussian"),

        load_model_data("all_predictions_testing_gaussian_laplace.npy",
                        "all_sigmas_testing_gaussian_laplace.npy",
                        "all_ground_truths_testing_gaussian_laplace.npy",
                        label="Gauss–Laplace", latent="gaussian", likelihood="laplace"),

        load_model_data("all_predictions_test_laplace_gaussian.npy",
                        "all_sigmas_test_laplace_gaussian.npy",
                        "all_ground_truths_test_laplace_gaussian.npy",
                        label="Laplace–Gauss", latent="laplace", likelihood="gaussian"),

        load_model_data("all_predictions_testing_laplace_laplace.npy",
                        "all_sigmas_testing_laplace_laplace.npy",
                        "all_ground_truths_testing_laplace_laplace.npy",
                        label="Laplace–Laplace", latent="laplace", likelihood="laplace"),
    ]

    visualizer = GroupedModelComparisonVisualizer(num_dims=NUM_DIMS)
    visualizer.fit_calibrators(val_model_data)
    visualizer.plot_grouped_cdfs(test_model_data, save_path="calibration_comparison.pdf")
