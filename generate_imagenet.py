# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import glob
import click
import pickle
from PIL import Image, ImageEnhance
from tqdm import tqdm
from tqdm.contrib import tzip

import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F

import dnnlib
from training import dataset as ds
from torch_utils import misc
from torch_utils import distributed as dist
from blurring import dct_2d, idct_2d
from blurring import block_noise, get_alpha_t


def blur_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=250, sigma_min=0.008, sigma_max=80, rho=7,
    truncation_sigma=0.9, truncation_t=0.93, up_scale=4, cfg_scale=1,
    s_block=0.15, s_noise=0.2, blur_sigma_max=3
):
    """
    truncation_sigma: Truncation point of noise schedule
    up_scale: Scale of upsampling, default 256/64=4
    cfg_scale: Scale of classifier-free guidance
    s_block: Scale of block noise addition
    s_noise: Scale of stochasticity in sampler
    """
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.S
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    idx_after_truncation = 0
    while t_steps[idx_after_truncation] >= truncation_sigma:
        idx_after_truncation += 1
    t_steps = t_steps[idx_after_truncation:]
    num_steps = len(t_steps)
    
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64)
    x_cur = None
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        
        if x_cur is None:
            if s_block > 0:
                x_cur = x_next + randn_like(x_next) * t_cur + s_block * block_noise(latents, randn_like, up_scale, device=latents.device) * t_cur
            else:
                x_cur = x_next + randn_like(x_next) * t_cur
        else:
            x_cur = x_next
            
        # Euler step.
        if cfg_scale > 1:
            denoised_cond = net(x_cur, t_cur, class_labels).to(torch.float64)
            denoised_uncond = net(x_cur, t_cur, None).to(torch.float64)
            denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
        else:
            denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
            
        if i == num_steps - 1:
            return denoised
        
        alpha_next = get_alpha_t(t_next, up_scale, latents.device, prob_length=truncation_t, blur_sigma_max=blur_sigma_max)
        alpha_cur = get_alpha_t(t_cur, up_scale, latents.device, prob_length=truncation_t, blur_sigma_max=blur_sigma_max)
        
        u_cur = dct_2d(x_cur, up_scale, norm='ortho')
        u_0 = dct_2d(denoised, up_scale, norm='ortho')
        d_cur = (u_cur - u_0) / t_cur
        
        gamma = (1 - s_noise**2)**0.5 * t_next / t_cur
        
        if s_block > 0:
            x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                + t_cur * (gamma * alpha_cur - alpha_next) * d_cur, up_scale, norm='ortho') \
                + s_noise * t_next * (randn_like(x_next) + s_block * block_noise(latents, randn_like, up_scale, device=latents.device))
        else:
            x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                + t_cur * (gamma * alpha_cur  - alpha_next) * d_cur, up_scale, norm='ortho') \
                + s_noise * t_next * randn_like(x_next)
                
        # Apply 2nd order correction.
        if i < num_steps - 1:
            if cfg_scale > 1:
                denoised_cond = net(x_next, t_next, class_labels).to(torch.float64)
                denoised_uncond = net(x_next, t_next, None).to(torch.float64)
                denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
            else:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                  
            u_next = dct_2d(x_next, up_scale, norm='ortho')
            u_0 = dct_2d(denoised, up_scale, norm='ortho')
            d_prime = (u_next - u_0) / t_next
            if s_block > 0:
                x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                    + t_cur * (gamma * alpha_cur - alpha_next) * (d_cur + d_prime) / 2, up_scale, norm='ortho') \
                    + s_noise * t_next * (randn_like(x_next) + s_block * block_noise(latents, randn_like, up_scale, device=latents.device))
            else:
                x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                    + t_cur * (gamma * alpha_cur - alpha_next) * (d_cur + d_prime) / 2, up_scale, norm='ortho') \
                    + s_noise * t_next * randn_like(x_next)
                    
    return x_next


def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=256, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=40, S_min=0.05, S_max=50, S_noise=1.003, cfg_scale=1
):
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        # Euler step.
        if cfg_scale > 1:
            denoised_cond = net(x_hat, t_hat, class_labels).to(torch.float64)
            denoised_uncond = net(x_hat, t_hat, None).to(torch.float64)
            denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
        else:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        
        # ## adding SDE
        # g_t = 0.05 * t_next
        # a_t = ((t_next**2 - g_t**2) / t_hat**2) ** 0.5

        # x_next = x_hat + (a_t - 1) * t_hat * d_cur + g_t * randn_like(x_next)

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (a_t - 1) * t_hat * (0.5 * d_cur + 0.5 * d_prime) + g_t * randn_like(x_next)
        
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            if cfg_scale > 1:
                denoised_cond = net(x_next, t_next, class_labels).to(torch.float64)
                denoised_uncond = net(x_next, t_next, None).to(torch.float64)
                denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
            else:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Image enhance
def img_enhance(img):
    enh_bri = ImageEnhance.Brightness(img)
    factor = 1.06
    return enh_bri.enhance(factor)


#----------------------------------------------------------------------------
# Sample saver

def save_samples(images, batch_seeds, out_dir):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for seed, image_np in zip(batch_seeds, images_np):
        image_dir = os.path.join(out_dir, f'{seed - seed % 1000:06d}')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'{seed:06d}.png')
        if image_np.shape[2] == 1:
            img_enhance(Image.fromarray(image_np[:, :, 0], 'L')).save(image_path)
        else:
            img_enhance(Image.fromarray(image_np, 'RGB')).save(image_path)

#----------------------------------------------------------------------------

@click.command()
@click.option('--indir',                     help='Input directory for only-second-stage sampler', metavar='DIR',     type=str)
@click.option('--outdir',                    help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                     help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='10000-59999', show_default=True)
@click.option('--subdirs',                   help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',        help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',   help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--sampler_stages',            help='Which stage to conduct sampler', metavar='first|second|both',      type=click.Choice(['first', 'second', 'both']), default='both')

# first stage sampler config
@click.option('--network_first',             help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--num_steps_first',           help='Number of sampling steps for first stage', metavar='INT',          type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--sigma_min_first',           help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_first',           help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho_first',                 help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_first',           help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--S_churn', 'S_churn_first',  help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=40, show_default=True)
@click.option('--S_min', 'S_min_first',      help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0.05, show_default=True)
@click.option('--S_max', 'S_max_first',      help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=50, show_default=True)
@click.option('--S_noise', 'S_noise_first',  help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1.003, show_default=True)

# second stage sampler config
@click.option('--network_second',            help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--num_steps_second',          help='Number of sampling steps for second stage', metavar='INT',         type=click.IntRange(min=1), default=200, show_default=True)
@click.option('--sigma_min_second',          help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_second',          help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--blur_sigma_max_second',     help='Maximum sigma of blurring schedule', metavar='FLOAT',              type=click.FloatRange(min=0), default=3, show_default=True)
@click.option('--rho_second',                help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=100, show_default=True)
@click.option('--cfg_scale_second',          help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=3.5, show_default=True)
@click.option('--up_scale_second',           help='Scale of upsampling, default 256/64=4', metavar='FLOAT',           type=click.IntRange(min=2), default=4, show_default=True)
@click.option('--truncation_sigma_second',   help='Truncation point of noise schedule', metavar='FLOAT',              type=click.FloatRange(min=0, min_open=True), default=0.9, show_default=True)
@click.option('--truncation_t_second',       help='Truncation point of time schedule', metavar='FLOAT',               type=click.FloatRange(min=0, min_open=True), default=0.93, show_default=True)
@click.option('--s_block_second',            help='Strength of block noise addition', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.15, show_default=True)
@click.option('--s_noise_second',            help='Strength of stochasticity', metavar='FLOAT',                       type=click.FloatRange(min=0), default=0.2, show_default=True)

def main(outdir, subdirs, seeds, class_idx, max_batch_size, sampler_stages, 
         network_first=None, network_second=None, indir=None,
         device=torch.device('cuda'), **sampler_kwargs):
    
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    if sampler_stages in ['first', 'both']:
        dist.print0(f'Loading first stage network from "{network_first}"...')
        
        assert network_first.endswith('pkl') or network_first.endswith('pt'), "Unknown format of the ckpt filename"
        if network_first.endswith('.pkl'):
            with dnnlib.util.open_url(network_first, verbose=(dist.get_rank() == 0)) as f:
                net_first = pickle.load(f)['ema'].to(device)
        elif network_first.endswith('.pt'):
            data = torch.load(network_first, map_location=torch.device('cpu'))
            net_first = data['ema'].eval().to(device)
        
        first_stage_sampler_kwargs = {
            k[:-6]: v for k, v in sampler_kwargs.items() if k.endswith('_first') and v is not None
        }
    if sampler_stages in ['second', 'both']:
        dist.print0(f'Loading second stage network from "{network_second}"...')
        
        assert network_second.endswith('pkl') or network_second.endswith('pt'), "Unknown format of the ckpt filename"
        if network_second.endswith('.pkl'):
            with dnnlib.util.open_url(network_second, verbose=(dist.get_rank() == 0)) as f:
                net_second = pickle.load(f)['ema'].to(device)
        elif network_second.endswith('.pt'):
            data = torch.load(network_second, map_location=torch.device('cpu'))
            net_second = data['ema'].eval().to(device)
        
        second_stage_sampler_kwargs = {
            k[:-7]: v for k, v in sampler_kwargs.items() if k.endswith('_second') and v is not None
        }
    
    if sampler_stages == 'second':
        # Preload for only-second-stage sampling.
        dist.print0(f'Preloading first stage samples from "{indir}"...')
        preload_images = []
        for batch_seeds in rank_batches:
            image_paths = [os.path.join(indir, f'{seed - seed % 1000:06d}', f'{seed:06d}.png') for seed in batch_seeds]
            batch_images = [np.array(Image.open(path)) for path in image_paths]
            batch_images = [image[np.newaxis, :, :] if image.ndim == 2 else image.transpose(2, 0, 1) for image in batch_images]
            batch_images = np.concatenate([image[np.newaxis, ...] for image in batch_images], axis=0)
            preload_images.append(batch_images)
            
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    dist.print0('first stage config:', first_stage_sampler_kwargs)
    dist.print0('second stage config:', second_stage_sampler_kwargs)
    
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for i, batch_seeds in tzip(range(len(rank_batches)), rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        # Pick labels.
        class_labels = None
        label_dim = (net_first or net_second).label_dim
        if label_dim:
            class_labels = torch.eye(label_dim, device=device)[batch_seeds % label_dim]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1
        
        if sampler_stages in ['first', 'both']:
            # First stage generation.
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net_first.img_channels, net_first.img_resolution, net_first.img_resolution], device=device)
            images = edm_sampler(net_first, latents, class_labels, randn_like=rnd.randn_like, **first_stage_sampler_kwargs)
        else:
            images = torch.tensor(preload_images[i], device=device, dtype=torch.float64) / 127.5 - 1
        
        if sampler_stages == 'first':
            # Save outputs
            save_samples(images, batch_seeds, outdir)
            continue
        else:
            # Upsample for second stage generation.
            images = F.interpolate(images, 256)
            
        if sampler_stages in ['second', 'both']:
            # Second stage generation.
            images = blur_sampler(net_second, images, class_labels, randn_like=rnd.randn_like, **second_stage_sampler_kwargs)
            
            save_samples(images, batch_seeds, outdir)
        
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------