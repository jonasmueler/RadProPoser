import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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


def test_calibration_with_synthetic_data(N=1000, D=78, true_variance=0.1):
    torch.manual_seed(42)
    mu = torch.randn(N, D)
    var = torch.full((N, D), true_variance)
    gt = mu + torch.randn_like(mu) * torch.sqrt(var)

    cal_error, sharpness, (expected_p, observed_p) = compute_calibration_and_sharpness(mu, var*4, gt)

    print(f"✅ Calibration error: {cal_error:.6f} (should be close to 0)")
    print(f"✅ Sharpness: {sharpness:.6f} (should be close to {true_variance})")

    plt.figure(figsize=(6, 6))
    plt.plot(expected_p, observed_p, label='Model', marker='o')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel("Expected Confidence Level")
    plt.ylabel("Observed Frequency")
    plt.title(f"Calibration Curve\nCalibration Error: {cal_error:.4f}, Sharpness: {sharpness:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run locally
    test_calibration_with_synthetic_data()