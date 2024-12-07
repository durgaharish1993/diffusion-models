import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to create a 3D Gaussian distribution and visualize it
def visualize_gaussian(mean, cov_matrix, num_samples):
    # Create the multivariate normal distribution
    mvn = torch.distributions.MultivariateNormal(loc=torch.tensor(mean), covariance_matrix=torch.tensor(cov_matrix))

    # Generate samples
    samples = mvn.sample((num_samples,))

    # Convert samples to NumPy for visualization
    samples_np = samples.numpy()

    # Create a 3D plot for the samples
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the samples
    ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], s=10, alpha=0.5, color='b')

    # Set labels
    ax.set_title('3D Scatter Plot of Gaussian Distribution Samples')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')

    # Create a meshgrid for the PDF
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    x1, x2 = np.meshgrid(x, y)

    # Calculate the PDF over the grid
    pos = torch.tensor(np.dstack((x1, x2, np.zeros_like(x1))))  # Z will be zero for the base layer
    pdf_values = mvn.log_prob(pos).exp().numpy()  # Calculate PDF values

    # Create a surface plot for the PDF
    ax.plot_surface(x1, x2, pdf_values, cmap='viridis', alpha=0.5)

    # Show the plot in Streamlit
    st.pyplot(fig)

# Streamlit app
st.title("3D Gaussian Distribution Visualization")

# User input for mean and covariance
mean = st.text_input("Mean (comma-separated, e.g., 0,0,0):", "0,0,0")
mean = list(map(float, mean.split(",")))

cov_matrix_input = st.text_area("Covariance Matrix (comma-separated, row-wise, e.g., 1,0.5,0.2;0.5,1,0.3;0.2,0.3,1):", 
                                  "1,0.5,0.2;0.5,1,0.3;0.2,0.3,1")
cov_matrix = [list(map(float, row.split(","))) for row in cov_matrix_input.split(";")]

num_samples = st.slider("Number of Samples:", min_value=100, max_value=5000, value=10000)

# Button to visualize
if st.button("Visualize"):
    visualize_gaussian(mean, cov_matrix, num_samples)
