import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML

from diffusion_utilities import ResidualConvBlock



from dataclasses import dataclass

@dataclass
class Config:
    weight_path = f"weights/model_trained.pth"



@dataclass
class Parameters:
    # hyperparameters
    # diffusion hyperparameters
    timesteps = 600
    beta1     = 1e-4
    beta2     = 0.02

    # network hyperparameters
    device    = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat    = 64 # 64 hidden dimension feature
    n_cfeat   = 5 # context vector is of size 5
    height    = 16 # 16x16 image
    in_channel = 3 

    save_dir  = './weights/'


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        
        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 256, 4,  4]
        
         # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1    = EmbedFC(1, 2*n_feat)
        self.timeembed2    = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample  
            nn.GroupNorm(8, 2 * n_feat), # normalize                       
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)       #[10, 256, 8, 8]
        down2 = self.down2(down1)   #[10, 256, 4, 4]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)
        
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)



class Utils:
    @staticmethod
    def norm_all(store, n_t, n_s):
        def unorm(x):
            # unity norm. results in range of [0,1]
            # assume x (h,w,3)
            xmax = x.max((0,1))
            xmin = x.min((0,1))
            return(x - xmin)/(xmax - xmin)

        # runs unity norm on all timesteps of all samples
        nstore = np.zeros_like(store)
        for t in range(n_t):
            for s in range(n_s):
                nstore[t,s] = unorm(store[t,s])
        return nstore

    @staticmethod
    def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
        ncols = n_sample//nrows
        sx_gen_store  = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
        nsx_gen_store = Utils.norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
        
        # create gif of images evolving over time, based on x_gen_store
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
        def animate_diff(i, store):
            print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
            plots = []
            for row in range(nrows):
                for col in range(ncols):
                    axs[row, col].clear()
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
            return plots
        ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
        plt.close()
        if save:
            ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
            print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
        return ani


class BaseScheduler:
    def scheduler(self,):
        pass 

class DiffusionScheduler(BaseScheduler):
    def __init__(self):
        self.scheduler()
        
    def scheduler(self):
        temp          = torch.linspace(0, 1, Parameters.timesteps + 1, device=Parameters.device)
        self.b_t      = (Parameters.beta2 - Parameters.beta1) * temp + Parameters.beta1
        self.a_t      = 1 - self.b_t
        self.ab_t     = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0]  = 1


class DiffusionSample:

    def __init__(self,scheduler : DiffusionScheduler):
        self.scheduler = scheduler

    def denoise_add_noise(self,x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)

        noise = self.scheduler.b_t.sqrt()[t] * z
        mean  = (x - pred_noise * ((1 -  self.scheduler.a_t[t]) / (1 - self.scheduler.ab_t[t]).sqrt())) / self.scheduler.a_t[t].sqrt()
        return mean + noise
    

    # sample using standard algorithm
    @torch.no_grad()
    def sample_ddpm(self, nn_model : nn.Module, n_sample : int, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        # (N, in_channels, H, W) :: Batch-Size, Input Channels, Height, Width
        samples = torch.randn(n_sample, Parameters.in_channel , Parameters.height, Parameters.height).to(Parameters.device)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in range(Parameters.timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / Parameters.timesteps])[:, None, None, None].to(Parameters.device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps     = nn_model(samples, t)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate ==0 or i==Parameters.timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

if __name__ == '__main__':
    # construct DDPM noise schedule



    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=Parameters.n_feat, n_cfeat=Parameters.n_cfeat, height=Parameters.height).to(Parameters.device)

    # load in model weights and set to eval mode
    nn_model.load_state_dict(torch.load(Config.weight_path, map_location=Parameters.device))
    nn_model.eval()
    print("Loaded in Model")

    # visualize samples
    plt.clf()
    diff_sample = DiffusionSample(scheduler=DiffusionScheduler())
    samples, intermediate_ddpm = diff_sample.sample_ddpm(nn_model = nn_model, n_sample = 48 )
    animation_ddpm = Utils.plot_sample(intermediate_ddpm,48,4,Parameters.save_dir, "ani_run", None, save=True)
    HTML(animation_ddpm.to_jshtml())