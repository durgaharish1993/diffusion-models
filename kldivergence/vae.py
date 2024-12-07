import torch.nn as nn
import torch 
from generate_data import GenerateGussainData
import torch
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal



class GaussianVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GaussianVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Mean and log variance for q(z|x)
        self.fc_mu    = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder now outputs mean and log variance for Gaussian reconstruction
        self.decoder_mean = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Mean output
        )

        self.decoder_logvar = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Log variance output
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon_mu = self.decoder_mean(z)
        recon_logvar = self.decoder_logvar(z)
        return recon_mu, recon_logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z)
        return recon_mu, recon_logvar, mu, logvar


def gaussian_vae_loss(recon_mu, recon_logvar, x, mu, logvar):
    """
    Compute the loss for the Gaussian VAE. 
    Args:
        recon_mu (torch.Tensor): Reconstructed mean from the decoder.
        recon_logvar (torch.Tensor): Reconstructed log variance from the decoder.
        x (torch.Tensor): Original input data.
        mu (torch.Tensor): Latent mean from the encoder.
        logvar (torch.Tensor): Latent log variance from the encoder.
    Returns:
        loss (torch.Tensor): Total loss combining reconstruction and KL divergence.
    """
    # Gaussian log-likelihood for reconstruction: log p(x|z)
    recon_loss = 0.5 * torch.sum(recon_logvar + torch.exp(-recon_logvar) * (x - recon_mu)**2)

    # KL Divergence loss: D_KL(q(z|x) || p(z)), where p(z) is a standard Gaussian
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss



def train_vae(model, data, epochs, batch_size, optimizer):
    """
    Train the VAE model.
    Args:
        model (GaussianVAE): VAE model.
        data (torch.Tensor): Training data.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
    """
    dataset   = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, in dataloader:
            optimizer.zero_grad()
            recon_mu, recon_logvar, mu, logvar = model(batch_data)
            loss = gaussian_vae_loss(recon_mu, recon_logvar, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")



def generate_data_from_vae(model, n_samples, latent_dim):
    """
    Generate data by sampling from the latent space of the VAE and using the decoder.
    Args:
        model (GaussianVAE): Trained VAE model.
        n_samples (int): Number of data points to generate.
        latent_dim (int): Dimensionality of the latent space.
    Returns:
        generated_data (torch.Tensor): Generated data from the decoder.
    """
    # Sample latent points from standard normal distribution (p(z))
    z_samples = torch.randn(n_samples, latent_dim)
    
    # Use the decoder to generate data from these latent points
    with torch.no_grad():
        recon_mu, recon_logvar = model.decode(z_samples)
    
    # Use the predicted mean (recon_mu) as the generated data
    return recon_mu

def compare_distributions(original_data, generated_data):
    """
    Compare the distributions of the original and generated data using visual comparison
    and KL divergence.
    Args:
        original_data (torch.Tensor): Original input data sampled from a multivariate Gaussian.
        generated_data (torch.Tensor): Data generated from the VAE decoder.
    """
    # Visual comparison using scatter plot
    plt.figure(figsize=(10, 5))

    # Plot original data
    plt.subplot(1, 2, 1)
    plt.scatter(original_data[:, 0].numpy(), original_data[:, 1].numpy(), color='blue', alpha=0.5)
    plt.title('Original Data')
    plt.xlabel('X1')
    plt.ylabel('X2')

    # Plot generated data
    plt.subplot(1, 2, 2)
    plt.scatter(generated_data[:, 0].numpy(), generated_data[:, 1].numpy(), color='red', alpha=0.5)
    plt.title('Generated Data')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.show()

    # Calculate KL divergence between the two distributions
    original_mean = torch.mean(original_data, dim=0)
    original_cov = torch.cov(original_data.T)

    generated_mean = torch.mean(generated_data, dim=0)
    generated_cov = torch.cov(generated_data.T)

    original_dist = MultivariateNormal(original_mean, original_cov)
    generated_dist = MultivariateNormal(generated_mean, generated_cov)

    kl_div = kl_divergence(generated_dist, original_dist)

    print(f"KL Divergence between original and generated distributions: {kl_div.item():.4f}")



gen_obj = GenerateGussainData(n_samples=100000)
data = gen_obj.generate_data()
# Instantiate model and optimizer
input_dim = gen_obj.dim
latent_dim = gen_obj.latent_dim
# vae_model = GaussianVAE(input_dim=input_dim, latent_dim=latent_dim)
# optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)

# # Train the VAE
# train_vae(vae_model, data, epochs=100, batch_size=64, optimizer=optimizer)

# # Example usage:
# # Assume VAE is trained and data is already generated

# # Generate data from the trained VAE model
# latent_dim = 2
# generated_data = generate_data_from_vae(vae_model, n_samples=1000, latent_dim=latent_dim)

# Compare distributions
noise = gen_obj.generate_multivariate_gaussian_data(gen_obj.n_samples, 2, [2.0,2.0], [[0.1, 0],[0,0.1]])
compare_distributions(data, data + noise)




