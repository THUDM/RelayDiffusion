# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
import scipy.stats as st
from torch_utils import persistence

from blurring import block_noise, dct_2d, idct_2d

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

# def blocked_noise(ref_x, block_size=1, scale=1, device=None):
#     g_noise = torch.randn_like(ref_x, device=device) * scale
#     if block_size == 1:
#         return g_noise
    
#     blk_noise = torch.zeros_like(ref_x, device=device)
#     for px in range(block_size):
#         for py in range(block_size):
#             blk_noise += torch.roll(g_noise, shifts=(px, py), dims=(-2, -1))
            
#     blk_noise = blk_noise / block_size # to maintain the same std on each pixel
    
#     return blk_noise

def DCTBlur(x, patch_size, blur_sigmas, min_scale, device):
    blur_sigmas = torch.as_tensor(blur_sigmas).to(device)
    freqs = torch.pi * torch.linspace(0, patch_size-1, patch_size).to(device) / patch_size
    frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    t = blur_sigmas ** 2 / 2
    
    dct_coefs = dct_2d(x, patch_size, norm='ortho')
    scale = x.shape[-1] // patch_size
    dct_coefs = dct_coefs * (torch.exp(-frequencies_squared.repeat(scale,scale) * t) * (1 - min_scale) + min_scale)
    return idct_2d(dct_coefs, patch_size, norm='ortho')


@persistence.persistent_class
class BlurLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, up_scale=4, prob_length=0.93, blur_sigma_max=3, min_scale=0.001, block_scale=0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        self.up_scale = up_scale
        self.prob_length = prob_length
        self.blur_sigma_max = blur_sigma_max
        self.min_scale = min_scale
      
        self.block_scale = block_scale
        
    def __call__(self, net, images, labels=None, augment_pipe=None, truncate_sigma=7.5e-3):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1])
        truncate_p = st.norm.cdf((np.log(truncate_sigma) - self.P_mean) / self.P_std) * self.prob_length # truncate for very little sigma to stabilize training
        rnd_uniform = torch.clamp(rnd_uniform, min=truncate_p)  # sigma = torch.clamp(sigma, min=0.0075)
        
        blur_sigmas = self.blur_sigma_max * torch.sin(rnd_uniform * torch.pi / 2) ** 2
        
        rnd_interval_uniform = rnd_uniform * self.prob_length
        rnd_interval_normal = st.norm.ppf(rnd_interval_uniform)
        rnd_interval_normal = torch.tensor(rnd_interval_normal, device=images.device)
        
        sigma = (rnd_interval_normal * self.P_std + self.P_mean).exp()
        
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        weight = torch.clamp(weight, max=100000)
        
        y_blurred = DCTBlur(y, self.up_scale, blur_sigmas, self.min_scale, images.device)
        n = torch.randn_like(y_blurred) * sigma
        if self.block_scale > 0:
            n = torch.randn_like(y_blurred) * sigma + block_noise(y_blurred, block_size=self.up_scale, device=y_blurred.device) * sigma * self.block_scale
            
        D_yn = net(y_blurred + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
        