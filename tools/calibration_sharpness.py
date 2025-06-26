import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Optional, Tuple, List
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy.stats import norm, gaussian_kde

def compute_calibration_and_sharpness(mu, var, gt, num_bins=50):
    """
    Compute calibration error based on Kuleshov et al. (Accurate Uncertainties for Deep Learning Using Calibrated Regression).
    """
    mu = mu.detach().cpu().view(-1).numpy()
    var = var.detach().cpu().view(-1).numpy()
    gt = gt.detach().cpu().view(-1).numpy()
    sigma = np.sqrt(var)

    # Step 1: Compute true CDF probability p_i = P(Y <= y_i | mu_i, sigma_i)
    cdf_values = norm.cdf(gt, loc=mu, scale=sigma)

    # Step 2: Compare expected vs empirical CDF over fixed confidence levels
    expected_p = np.linspace(0.01, 0.99, num_bins)
    observed_p = [(cdf_values <= p).mean() for p in expected_p]

    # Step 3: Calibration error
    cal_error = np.mean((np.array(observed_p) - expected_p) ** 2)
    sharpness = var.mean()

    return cal_error, sharpness, (expected_p, observed_p)




def plot_calibration_coverage(
    mu_list: List[torch.Tensor],
    var_list: List[torch.Tensor],
    gt: torch.Tensor,
    model_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    num_bins: int = 50,
    ax: Optional[plt.Axes] = None,
    fill_alpha: float = 0.25,
    font_scale: float = 4,
) -> None:
    """
    Plot predicted-vs-empirical coverage for multiple models.

    Args:
        mu_list, var_list: Lists of tensors with shape (P, N, D) or (N, D).
        gt: Ground-truth tensor, shape (P, N, D) or (N, D).
        model_labels: Labels for legend (optional).
        colors: List of line colors (optional).
        fill_alpha: Opacity of spread band.
        font_scale: Global scaling factor for all font sizes.
    """

    # Set global font properties
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

    if not isinstance(mu_list, list):
        mu_list = [mu_list]
        var_list = [var_list]

    M = len(mu_list)
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(M)]
    if colors is None:
        colors = [None] * M

    if gt.dim() == 2:
        gt = gt.unsqueeze(0)

    expected_p = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for i, (mu, var, label, color) in enumerate(zip(mu_list, var_list, model_labels, colors)):
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
            var = var.unsqueeze(0)
        P = mu.size(0)

        observed_p_all = []
        for p in range(P):
            mu_p = mu[p].detach().cpu().view(-1).numpy()
            var_p = var[p].detach().cpu().view(-1).numpy()
            gt_p = gt[p].detach().cpu().view(-1).numpy()
            sigma_p = np.sqrt(var_p)
            cdf_values = norm.cdf(gt_p, loc=mu_p, scale=sigma_p)
            observed_p = [(cdf_values <= q).mean() for q in expected_p]
            observed_p_all.append(observed_p)

        observed_p_all = np.array(observed_p_all)
        mean_observed = observed_p_all.mean(axis=0)
        std_observed = observed_p_all.std(axis=0)

        line = ax.plot(expected_p, mean_observed, label=label, linewidth=2, color=color)[0]
        fill_color = line.get_color()

        ax.fill_between(
            expected_p,
            np.clip(mean_observed - 2 * std_observed, 0, 1),
            np.clip(mean_observed + 2 * std_observed, 0, 1),
            color=fill_color,
            alpha=fill_alpha,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Confidence Level")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.legend(frameon=False)
    #ax.set_title("Calibration Curve (±2 std spread)")

    # Remove duplicate 0.0 tick label on y-axis if it overlaps with x-axis
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    if 0.0 in xticks and 0.0 in yticks:
        yticks_cleaned = [y for y in yticks if not np.isclose(y, 0.0)]
        ax.set_yticks(yticks_cleaned)





def plot_predicted_quantile_distributions_only(
    mu_list: List[torch.Tensor],
    var_list: List[torch.Tensor],
    gt: torch.Tensor,
    model_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    font_scale: float = 4,
    num_bins: int = 50,
    show_ideal: bool = True,
) -> None:
    """
    Plot predicted quantile distributions (PIT histograms) for multiple models, using raw frequencies.

    Args:
        mu_list, var_list: Lists of tensors with shape (P, N, D) or (N, D).
        gt: Ground-truth tensor, shape (P, N, D) or (N, D).
        model_labels: Optional model names.
        colors: Optional list of colors per model.
        font_scale: Global font scale.
        num_bins: Number of bins for histogram.
        show_ideal: Whether to draw reference uniform frequency line.
    """
    import matplotlib.pyplot as plt

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

    if not isinstance(mu_list, list):
        mu_list = [mu_list]
        var_list = [var_list]

    if gt.dim() == 2:
        gt = gt.unsqueeze(0)

    M = len(mu_list)
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(M)]
    if colors is None:
        from matplotlib.cm import tab10
        colors = [tab10(i) for i in range(M)]

    plt.figure(figsize=(10, 5))

    for i, (mu, var) in enumerate(zip(mu_list, var_list)):
        if mu.dim() == 2:
            mu, var = mu.unsqueeze(0), var.unsqueeze(0)

        pred_q_all = []
        for p in range(mu.size(0)):
            mu_p = mu[p].detach().cpu().reshape(-1).numpy()
            var_p = var[p].detach().cpu().reshape(-1).numpy()
            gt_p = gt[p].detach().cpu().reshape(-1).numpy()

            sigma_p = np.sqrt(np.clip(var_p, 1e-9, None))  # Avoid sqrt of negative
            pred_q = norm.cdf(gt_p, loc=mu_p, scale=sigma_p)
            pred_q_all.extend(np.clip(pred_q, 0, 1))  # Clamp within [0, 1]

        pred_q_all = np.array(pred_q_all)

        # Plot raw frequency histogram
        plt.hist(
            pred_q_all,
            bins=num_bins,
            range=(0, 1),
            density=False,
            alpha=0.6,
            label=model_labels[i],
            color=colors[i],
            edgecolor="black",
        )

        # Plot ideal uniform line (same frequency in each bin)
        if show_ideal:
            total_qs = len(pred_q_all)
            ideal_freq = total_qs / num_bins
            plt.axhline(
                ideal_freq,
                linestyle=":",
                color="k",
                linewidth=1.5,
                label="Ideal Uniform" if i == 0 else None,
            )

    plt.title("Predicted Quantile Distributions (PIT Histogram)")
    plt.xlabel("Quantile (CDF of true target under predicted distribution)")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()




def test_calibration_with_synthetic_data(N: int = 1000, D: int = 78, true_variance: float = 0.1):
    """
    Simulates predictions from multiple models across 3 participants and plots the calibration curves.
    """
    torch.manual_seed(42)
    num_participants = 3

    # True latent ground-truth distribution
    mu_true = torch.randn(num_participants, N, D)
    base_var = torch.full((N, D), true_variance)
    gt = mu_true + torch.randn_like(mu_true) * torch.sqrt(base_var)

    mu_list = []
    var_list = []
    labels = ["Under-confident", "Over-confident"]
    colors = ["tab:blue", "tab:orange"]

    # Model 1: Under-confident (variance too large)
    mu1 = mu_true + 0.05 * torch.randn_like(mu_true)
    var1 = base_var.unsqueeze(0).repeat(num_participants, 1, 1) * 2.0
    mu_list.append(mu1)
    var_list.append(var1)

    # Model 2: Over-confident (variance too small)
    mu2 = mu_true + 0.05 * torch.randn_like(mu_true)
    var2 = base_var.unsqueeze(0).repeat(num_participants, 1, 1) * 0.25
    mu_list.append(mu2)
    var_list.append(var2)

    plot_calibration_coverage(mu_list, var_list, gt, model_labels=labels, colors=colors)
    plt.tight_layout()
    plt.show()


def test_calibration_with_synthetic_data_and_marginals(
    N: int = 1000,
    D: int = 78,
    true_variance: float = 0.1,
    font_scale: float = 4,
    num_bins: int = 50,
):
    """
    Simulates predictions from multiple models across 3 participants and sequentially plots:
    1. Calibration coverage
    2. Predicted quantile distribution
    3. Observed quantile distribution
    """
    torch.manual_seed(42)
    num_participants = 3

    # True latent ground-truth distribution
    mu_true = torch.randn(num_participants, N, D)
    base_var = torch.full((N, D), true_variance)
    gt = mu_true + torch.randn_like(mu_true) * torch.sqrt(base_var)

    mu_list = []
    var_list = []
    labels = ["Under-confident", "Over-confident"]
    colors = ["tab:blue", "tab:orange"]

    # Model 1: Under-confident
    mu1 = mu_true + 0.05 * torch.randn_like(mu_true)
    var1 = base_var.unsqueeze(0).repeat(num_participants, 1, 1) * 2.0
    mu_list.append(mu1)
    var_list.append(var1)

    # Model 2: Over-confident
    mu2 = mu_true + 0.05 * torch.randn_like(mu_true)
    var2 = base_var.unsqueeze(0).repeat(num_participants, 1, 1) * 0.25
    mu_list.append(mu2)
    var_list.append(var2)

    # ─── Plot 1: Calibration coverage ─────────────────────────────
    plot_calibration_coverage(
        mu_list,
        var_list,
        gt,
        model_labels=labels,
        colors=colors,
        font_scale=font_scale,
        num_bins=num_bins,
    )
    plt.tight_layout()
    plt.show()

    # ─── Plot 2 & 3: Marginal quantile distributions ──────────────
    plot_predicted_quantile_distributions_only(
        mu_list,
        var_list,
        gt,
        model_labels=labels,
        colors=colors,
        font_scale=font_scale,
        num_bins=num_bins,
    )




if __name__ == "__main__":
    # Run locally
    test_calibration_with_synthetic_data_and_marginals()


