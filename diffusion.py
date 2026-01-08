import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class Diffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, img_size: int = 32, img_channels:int = 3, device = torch.device("cpu")) -> None:
        super().__init__()
        self.device = device
        self.timesteps =  timesteps
        self.betas = torch.linspace(start = 1e-4, end = 0.02, steps = self.timesteps).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.img_size = img_size
        self.img_channels = img_channels
        

    def noise_images(self, x, t):
        eps = torch.randn_like(x)
        noised_img = torch.sqrt(self.alphas_cumprod[t][:, None, None, None])*x + torch.sqrt(1 - self.alphas_cumprod[t][:, None, None, None])*eps
        return noised_img, eps

    def sample_t(self, n):
        return torch.randint(low = 1, high = self.timesteps, size = (n, )).to(self.device)

    def sample_img(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size)).to(self.device)
            for i in range(self.timesteps-1, 0, -1):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_eps = model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alphas_cumprod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)

                x =  (1/ torch.sqrt(alpha)) * (x - ((1-alpha) / torch.sqrt(1 - alpha_hat)) * predicted_eps) + torch.sqrt(beta)*z
                x = x.clamp(-1, 1)
            model.train()
            x = (x+1)/2
            x = (x*255).type(torch.uint8)
            return x