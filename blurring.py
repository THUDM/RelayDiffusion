"""Taken from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
Some modifications have been made to work with newer versions of Pytorch"""

import torch
import numpy as np
import scipy.stats as st

def rearrange(image, patch_size):
    """
    rearrange [B x C x hs x ws] images into [Bhw x C x s x s] patches
    """
    B, C, H, W = image.shape
    return image.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_size, patch_size)

def reverse_rearrange(image, image_size):
    """
    recover from [Bhw x C x s x s] patches to [B x C x hs x ws] images
    """
    _, C, _, patch_size = image.shape
    return image.reshape(-1, image_size // patch_size, image_size // patch_size, C, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(-1, C, image_size, image_size)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
   
    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, size, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    origin_size = x.shape[-1]
    if origin_size > size:
        x = rearrange(x, size)
        
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X2 = X2.transpose(-1, -2)
    
    if origin_size > size:
        X2 = reverse_rearrange(X2, origin_size)
    
    return X2

def idct_2d(X, size, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    origin_size = X.shape[-1]
    if origin_size > size:
        X = rearrange(X, size)

    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x2 = x2.transpose(-1, -2)

    if origin_size > size:
        x2 = reverse_rearrange(x2, origin_size)

    return x2


def block_noise(ref_x, randn_like=torch.randn_like, block_size=1, device=None):
    """
    build block noise
    """
    g_noise = randn_like(ref_x)
    if block_size == 1:
        return g_noise
    
    blk_noise = torch.zeros_like(ref_x, device=device)
    for px in range(block_size):
        for py in range(block_size):
            blk_noise += torch.roll(g_noise, shifts=(px, py), dims=(-2, -1))
            
    blk_noise = blk_noise / block_size # to maintain the same std on each pixel
    
    return blk_noise

def get_alpha_t(t, patch_size, device, repeat_dims=64, prob_length=0.93, blur_sigma_max=3, min_scale=0.001):
    """
    build blurring matrix on time t
    """
    P_mean, P_std = -1.2, 1.2
    t = st.norm.cdf((np.log(t.cpu()) - P_mean)/ P_std) / prob_length
    t = torch.tensor(t, device=device)
    
    blur_sigmas = blur_sigma_max * torch.sin(t * torch.pi / 2)**2
    blur_sigmas = blur_sigmas.to(device)
    blur_ts = blur_sigmas**2 / 2
    
    freqs = torch.pi * torch.linspace(0, patch_size - 1, patch_size).to(device) / patch_size
    freqs_squared = freqs[:, None]**2 + freqs[None, :]**2
    
    return torch.exp(-freqs_squared.repeat(repeat_dims, repeat_dims) * blur_ts) * (1 - min_scale) + min_scale