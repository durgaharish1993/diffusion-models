import torch
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
torch.manual_seed(0)

# Define parameters for the multivariate normal distribution
mean = torch.tensor([0.0, 0.0])  # Mean
cov_matrix = torch.tensor([[1.0, 0.5],  # Covariance matrix
                           [0.5, 1.0]])

# Create the multivariate normal distribution
mvn = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)

# Generate samples
num_samples = 1000
samples = mvn.sample((num_samples,))
print(samples)
# Compute the covariance matrix of the samples
sample_mean = samples.mean(dim=0)
cov_samples = torch.cov(samples.T)  # Transpose to get correct shape
print("Computed Covariance Matrix:\n", cov_samples)

# Convert samples to NumPy for visualization
samples_np = samples.numpy()

# Plot the samples in a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(samples_np[:, 0], samples_np[:, 1], s=10, alpha=0.5, label='Samples')

# Create a meshgrid for the PDF
x1, x2 = np.mgrid[-3:3:.01, -3:3:.01]  # Create a grid of x1 and x2 values
grid = torch.tensor(np.dstack((x1, x2)), dtype=torch.float32)

# Calculate the PDF over the grid
pdf_values = torch.exp(mvn.log_prob(grid)).numpy()
print(pdf_values)
# Plot the PDF as a contour plot
plt.contourf(x1, x2, pdf_values, levels=20, cmap='viridis', alpha=0.5)
plt.colorbar(label='Density')

# Add labels and title
plt.title('Scatter Plot with Contours of Multivariate Normal Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
