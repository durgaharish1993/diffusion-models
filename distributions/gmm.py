import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate synthetic data
def generate_data(num_samples, num_clusters, means, covariances):
    samples = []
    for i in range(num_clusters):
        cluster_samples = np.random.multivariate_normal(mean=means[i], cov=covariances[i], size=num_samples)
        samples.append(cluster_samples)
    return np.vstack(samples)

# Parameters for synthetic data
num_samples = 300
num_clusters = 3
means = [(-3, -3), (3, 3), (3, -3)]
covariances = [np.eye(2), np.eye(2), np.eye(2)]

# Generate synthetic dataset
data = generate_data(num_samples, num_clusters, means, covariances)

# Convert to PyTorch tensor
data_tensor = torch.tensor(data, dtype=torch.float32)

# Gaussian Mixture Model implementation
class GMM:
    def __init__(self, n_components, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.means = None
        self.covariances = None
        self.weights = None

    def fit(self, data):
        n_samples, n_features = data.shape
        # Initialize parameters
        self.means = data[torch.randperm(n_samples)[:self.n_components]]
        self.covariances = torch.stack([torch.eye(n_features) for _ in range(self.n_components)])
        self.weights = torch.ones(self.n_components) / self.n_components
        
        for _ in range(self.n_iter):
            # E-step
            responsibilities = self.e_step(data)
            # M-step
            self.m_step(data, responsibilities)

    def e_step(self, data):
        n_samples = data.shape[0]
        likelihoods = torch.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights[k] * self.multivariate_normal(data, self.means[k], self.covariances[k])
        
        # Compute responsibilities
        responsibilities = likelihoods / likelihoods.sum(dim=1, keepdim=True)
        return responsibilities

    def m_step(self, data, responsibilities):
        n_samples = data.shape[0]
        
        # Update weights
        effective_n = responsibilities.sum(dim=0)
        self.weights = effective_n / n_samples
        
        # Update means
        self.means = (responsibilities.T @ data) / effective_n.unsqueeze(1)
        
        # Update covariances
        for k in range(self.n_components):
            diff = data - self.means[k]
            self.covariances[k] = (responsibilities[:, k].unsqueeze(1) * diff).T @ diff / effective_n[k]
        
    def multivariate_normal(self, data, mean, covariance):
        n_features = mean.shape[0]
        diff = data - mean
        precision = torch.linalg.inv(covariance)
        exponent = -0.5 * (diff @ precision @ diff.T).diag()
        return (1 / ((2 * np.pi) ** (n_features / 2) * torch.sqrt(torch.det(covariance)))) * torch.exp(exponent)

    def predict(self, data):
        responsibilities = self.e_step(data)
        return responsibilities.argmax(dim=1)

# Fit GMM to data
gmm = GMM(n_components=num_clusters, n_iter=100)
gmm.fit(data_tensor)

# Make predictions
predictions = gmm.predict(data_tensor)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], c=predictions.numpy(), cmap='viridis', s=20, alpha=0.5)
plt.scatter(gmm.means[:, 0].detach().numpy(), gmm.means[:, 1].detach().numpy(), c='red', s=200, marker='X')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()
