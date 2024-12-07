import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define image dimensions and device
image_size = 28  # Example: 28x28 image (for simplicity)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy energy function (for demonstration purposes)
class EnergyFunction(nn.Module):
    def __init__(self):
        super(EnergyFunction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_size * image_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image to a vector
        energy = self.fc(x)
        return energy

# Langevin Dynamics function
def langevin_dynamics(init_image, energy_model, steps=200, step_size=0.01, noise_scale=0.005):
    image = init_image.clone().detach().requires_grad_(True).to(device)
    for i in range(steps):
        print("step number ::", i)
        # Calculate energy (negative log-likelihood) and its gradient
        energy = energy_model(image)
        energy.backward()

        # Langevin dynamics update: x = x - (step_size/2) * grad(x) + noise
        image.data -= step_size * image.grad / 2
        image.data += noise_scale * torch.randn_like(image)

        # Zero the gradients for the next step
        image.grad.zero_()

    return image

# Initialize the model and dummy image
energy_model = EnergyFunction().to(device)
initial_image = torch.randn(1, 1, image_size, image_size).to(device)  # Start from random noise

# Run Langevin Dynamics
generated_image = langevin_dynamics(initial_image, energy_model, steps=10000, step_size=0.1, noise_scale=0.01)

# Plot the generated image
generated_image_np = generated_image.detach().cpu().numpy().squeeze()
plt.imshow(generated_image_np, cmap='gray')
plt.title("Generated Image via Langevin Dynamics")
plt.show()
