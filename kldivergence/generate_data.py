import torch
from dataclasses import dataclass

class GenerateGussainData(object):
    def __init__(self,n_samples):
        self.dim = 2 
        self.n_samples = n_samples 
        self.latent_dim = 2 

    @staticmethod
    def generate_multivariate_gaussian_data(n_samples, dim, mean, cov):
        """
        Generate multivariate Gaussian data.
        Args:
            n_samples (int): Number of data points to generate.
            dim (int): Dimensionality of the data.
            mean (list): Mean vector of the Gaussian distribution.
            cov (list): Covariance matrix of the Gaussian distribution.
        Returns:
            data (torch.Tensor): Generated multivariate Gaussian data.
        """
        distribution = torch.distributions.MultivariateNormal(torch.tensor(mean), torch.tensor(cov))
        data = distribution.sample((n_samples,))
        return data

    def generate_data(self):
        # Example: Generate 1000 samples from a 2D Gaussian distribution
        n_samples = self.n_samples
        dim       =  self.dim 
        mean = [1.0, -1.0]
        cov  = [[1.0, 0.8], [0.8, 1.0]]  # Covariance matrix

        data = self.generate_multivariate_gaussian_data(n_samples, dim, mean, cov)
        
        return data 
    



