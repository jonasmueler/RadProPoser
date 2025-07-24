import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
b = 0.2      # Laplace scale parameter for a sharp peak
sigma = 0.2  # Gaussian standard deviation for comparable width
N = 400      # number of points

# Create a 1D grid for X
X = np.linspace(-1, 1, N)

# Compute PDFs
def laplace_pdf(x, b=1.0):
    return (1 / (2 * b)) * np.exp(-np.abs(x) / b)

def gaussian_pdf(x, sigma=1.0):
    return norm.pdf(x, scale=sigma)

Y_laplace = laplace_pdf(X, b)
Y_gaussian = gaussian_pdf(X, sigma)

# Choose a colormap
cmap = plt.get_cmap('viridis')
fill_color = cmap(0.6)

# Plotting and saving Laplace density
fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(X, Y_laplace, c=Y_laplace, cmap=cmap, s=10)
ax.fill_between(X, Y_laplace, color=fill_color, alpha=0.3)
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig('laplace_density.pdf', format='pdf')
plt.close(fig)

# Plotting and saving Gaussian density
fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(X, Y_gaussian, c=Y_gaussian, cmap=cmap, s=10)
ax.fill_between(X, Y_gaussian, color=fill_color, alpha=0.3)
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig('gaussian_density.pdf', format='pdf')
plt.close(fig)

print("Saved 'laplace_density.pdf' and 'gaussian_density.pdf' to current directory.")
