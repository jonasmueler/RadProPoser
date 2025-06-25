import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd

def plot_keypoint_calibration(mu, var, gt, method_name="Evidential", model_path=""):
    """
    Plots a calibration curve (expected vs observed confidence) for scalar keypoint coordinates.
    Each row of mu, var, gt should correspond to a sample of 78 keypoint coordinates (26 keypoints × 3D).
    """
    assert mu.shape == var.shape == gt.shape, "Shapes of mu, var, gt must match"
    assert mu.shape[1] == 78, "Expecting 78 coordinates (26 keypoints × 3)"

    # Flatten everything
    mu_flat = mu.flatten()
    sigma_flat = np.sqrt(var.flatten())
    gt_flat = gt.flatten()

    # Confidence levels to evaluate
    expected_p = np.linspace(0.01, 0.99, 50)
    observed_p = []

    for p in expected_p:
        z = norm.ppf(p)  # z-score for given confidence
        lower = mu_flat - z * sigma_flat
        upper = mu_flat + z * sigma_flat
        in_interval = (gt_flat >= lower) & (gt_flat <= upper)
        obs = in_interval.mean()
        observed_p.append(obs)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        "Expected Confidence": expected_p,
        "Observed Confidence": observed_p,
        "Method": method_name
    })

    # Plot
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=df, x="Expected Confidence", y="Observed Confidence", label=method_name)
    plt.plot([0, 1], [0, 1], 'k--', label="Ideal")
    plt.xlabel("Expected Confidence")
    plt.ylabel("Observed Confidence")
    plt.title("Keypoint Uncertainty Calibration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df

# --- Test with dummy data ---
if __name__ == "__main__":
    np.random.seed(42)

    B = 200  # batch size
    D = 78   # 26 keypoints × 3D

    # Generate dummy predictions and GT
    mu = np.random.normal(loc=0.0, scale=1.0, size=(B, D))          # predicted means
    var = np.random.uniform(low=0.05, high=0.5, size=(B, D))        # predicted variances
    gt = mu + np.random.normal(scale=np.sqrt(var))                  # generate GT from predicted dist

    # Plot calibration
    plot_keypoint_calibration(mu, var, gt, method_name="DummyModel")
