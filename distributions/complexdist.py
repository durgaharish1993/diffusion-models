import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

class ComplexDistribution:
    def __init__(self, weights, distributions):
        """
        Initialize the complex distribution using PyTorch.

        Parameters:
        weights (list): List of weights for each distribution in the mixture. Should sum to 1.
        distributions (list): List of PyTorch distributions.
        """
        self.weights = torch.tensor(weights, dtype=torch.float32)
        if not torch.isclose(self.weights.sum(), torch.tensor(1.0)):
            raise ValueError("The weights should sum up to 1.")
        
        self.distributions = distributions
        
        if len(self.weights) != len(self.distributions):
            raise ValueError("The number of weights must match the number of distributions.")

    def sample(self, size=1):
        """
        Generate samples from the mixture distribution.

        Parameters:
        size (int): Number of samples to generate.

        Returns:
        torch.Tensor: Tensor of samples.
        """
        samples = []
        for _ in range(size):
            # Choose a distribution index based on the weights
            idx = torch.multinomial(self.weights, num_samples=1).item()
            dist = self.distributions[idx]
            # Sample from the chosen distribution
            samples.append(dist.sample().item())
        return torch.tensor(samples)

    def pdf(self, x):
        """
        Calculate the probability density function (PDF) of the mixture distribution at point x.

        Parameters:
        x (torch.Tensor): Tensor of points where PDF needs to be evaluated.

        Returns:
        torch.Tensor: The value of the PDF at point(s) x.
        """
        total_pdf = torch.zeros_like(x, dtype=torch.float32)
        for weight, dist in zip(self.weights, self.distributions):
            total_pdf += weight * torch.exp(dist.log_prob(x))
        return total_pdf

# Define the mixture components: Normal, Exponential, and Uniform distributions
dist1 = dist.Normal(loc=0.0, scale=1.0)       # Standard Normal Distribution
dist2 = dist.Normal(loc=2.0, scale=0.4)  #dist.Exponential(rate=1.0)            # Exponential Distribution with lambda=1
dist3 = dist.Normal(loc=6.0, scale=1.0) #dist.Uniform(low=-3.0, high=3.0)      # Uniform Distribution between -2 and 2

# Define the weights for the mixture
weights = [0.01, 0.39, 0.6]  # These should sum to 1
min_value, max_value = -6,12
STEPS = 1000


# Create the complex distribution
complex_dist = ComplexDistribution(weights, [dist1, dist2, dist3])

# Example usage
size = 100000
samples = complex_dist.sample(size=size)          # Generate 1000 samples
x = torch.linspace(min_value, max_value, steps=STEPS)             # Points to evaluate the PDF
pdf_values = complex_dist.pdf(x)                  # Calculate PDF at the points

# Convert samples to NumPy for visualization
samples_np = samples.numpy()
x_np = x.numpy()
pdf_values_np = pdf_values.numpy()

# Plot the sampled data
plt.hist(samples_np, bins=30, density=True, alpha=0.6, color='g', label="Samples")
#Plot the PDF
plt.plot(x_np, pdf_values_np, label="PDF", color='r')
plt.title('Mixture of Normal, Exponential, and Uniform Distributions')
plt.legend()
plt.show()
