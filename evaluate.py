# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import time
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from training import dataset

from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

#----------------------------------------------------------------------------

def calculate_activations(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    s_feature_dim = 2023
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)
        
    spatial_features = []
    def get_spatial_features(module, input, output):
        spatial_features.append(output.to(torch.float64))
        return None
    detector_net.layers.mixed_6.conv.register_forward_hook(get_spatial_features)
        
    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    clean_rank_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)
    
    s_features, features, indices = [], [], []
    for i, (images, _labels) in enumerate(tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0))):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features.append(detector_net(images.to(device), **detector_kwargs).cpu().numpy().astype(np.float64))
        indices.append(clean_rank_batches[i].numpy())
        
        spatial_feature = spatial_features.pop()[:, :7, :, :]
        s_features.append(spatial_feature.reshape(spatial_feature.shape[0], -1).cpu().numpy().astype(np.float64))
        
    features = np.concatenate(features, axis=0)
    s_features = np.concatenate(s_features, axis=0)
    indices = np.concatenate(indices, axis=0)
    return features, s_features, indices, len(dataset_obj)

#----------------------------------------------------------------------------

def calculate_inception_stats_from_activations(
    activations, batch_size=64, device=torch.device('cuda')
):
    torch.distributed.barrier()
    
    data_num, feature_dim = activations.shape
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for i in range((data_num-1) // batch_size + 1):
        activations_batch = torch.tensor(activations[i*batch_size: (i+1)*batch_size], dtype=torch.float64, device=device)
        mu += activations_batch.sum(0)
        sigma += activations_batch.T @ activations_batch
    
    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= data_num
    sigma -= mu.ger(mu) * data_num
    sigma /= data_num - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

def _numpy_partition(arr, kth, **kwargs):
    # num_workers = min(cpu_count(), len(arr))
    num_workers = 4
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size
        
    return [np.partition(batch, kth=kth, **kwargs) for batch in batches]

    # with ThreadPool(num_workers) as pool:
    #     return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))

def _pairwise_distances(U, V):
    """
    Evaluate pairwise distances between two batches of feature vectors.
    """
    # Squared norms of each row in U and V.
    norm_u = np.square(U).sum(axis=1)
    norm_v = np.square(V).sum(axis=1)
    
    # norm_u as a column and norm_v as a row vectors.
    norm_u = norm_u.reshape(-1, 1)
    norm_v = norm_v.reshape(1, -1)
    
    # Pairwise squared Euclidean distances.
    D = norm_u - 2 * np.matmul(U, V.T) + norm_v
    D = np.max([D, np.zeros_like(D)], axis=0)
    
    return D

def _less_thans(batch_1, radii_1, batch_2, radii_2):
    D = _pairwise_distances(batch_1, batch_2)[..., None]
    batch_1_in = np.any(D <= radii_2, axis=1)
    batch_2_in = np.any(D <= radii_1[:, None], axis=0)
    return batch_1_in, batch_2_in

class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """
    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features):
        num_images = len(features)
        nhood_sizes = (3,)
        num_nhoods = len(nhood_sizes)
        
        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(nhood_sizes) + 1, dtype=np.int32)
        
        for begin1 in tqdm.tqdm(range(0, num_images, self.row_batch_size)):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]
            
            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]
                
                # Compute distances between batches.
                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = _pairwise_distances(row_batch, col_batch)
                
            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, nhood_sizes]
                    for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)
                ],
                axis=0,
            )
        return radii

    def evaluate_pr(self, features_1, radii_1, features_2, radii_2):
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in tqdm.tqdm(range(0, len(features_1), self.row_batch_size)):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = _less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )
            
def calculate_precision_recall_from_activations(activates_ref, activations_sample):
    estimator = ManifoldEstimator()
    radii_1 = estimator.manifold_radii(activates_ref)
    radii_2 = estimator.manifold_radii(activations_sample)
    pr = estimator.evaluate_pr(activates_ref, radii_1, activations_sample, radii_2)
    return float(pr[0]), float(pr[1])

#----------------------------------------------------------------------------

def calculate_inception_score_from_activations(activations, split_size=5000, device=torch.device('cuda')):
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)
    
    softmax_out = []
    softmax_batch_size = 512
    for i in tqdm.tqdm(range(0, len(activations), softmax_batch_size)):
        acts = activations[i : i + softmax_batch_size]
        feats = detector_net.output(torch.tensor(acts, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
        softmax_out.append(np.exp(feats) / np.sum(np.exp(feats), axis=-1)[..., None])
    preds = np.concatenate(softmax_out, axis=0)
    
    # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
    scores = []
    for i in range(0, len(preds), split_size):
        part = preds[i : i + split_size]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return float(np.mean(scores))

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

    \b
    # Compute dataset reference statistics
    python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
    """
    
#----------------------------------------------------------------------------

@main.command()
@click.option('-m', 'metrics',          help='Metrics to be calculated', metavar='STR',             type=click.Choice(['fid', 'sfid', 'is', 'pr']), multiple=True, required=True)
@click.option('--activations_sample',   help='Path to sample activations', metavar='PATH',          type=str, required=True)
@click.option('--activations_ref',      help='Path to ref activations', metavar='PATH',             type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)

def calc(metrics, activations_sample, activations_ref, batch):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    dist.print0(f'Loading sample activations from "{activations_sample}"...')
    dist.print0(f'Loading reference activations from "{activations_ref}"...')
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(activations_sample) as f:
            data = dict(np.load(f))
            feats_sample, s_feats_sample = data['feat'], data['s_feat']
        with dnnlib.util.open_url(activations_ref) as f:
            data = dict(np.load(f))
            feats_ref, s_feats_ref = data['feat'], data['s_feat']
            
        if 'fid' in metrics:
            mu_ref, sigma_ref = calculate_inception_stats_from_activations(feats_ref, batch_size=batch)
            mu, sigma = calculate_inception_stats_from_activations(feats_sample, batch_size=batch)
            fid = calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref)
            print(f'FID: {fid:g}')
        if 'sfid' in metrics:
            s_mu_ref, s_sigma_ref = calculate_inception_stats_from_activations(s_feats_ref, batch_size=batch)
            s_mu, s_sigma = calculate_inception_stats_from_activations(s_feats_sample, batch_size=batch)
            sfid = calculate_fid_from_inception_stats(s_mu, s_sigma, s_mu_ref, s_sigma_ref)
            print(f'sFID: {sfid:g}')
        if 'is' in metrics:
            inception_score = calculate_inception_score_from_activations(feats_sample)
            print(f'Inception Score: {inception_score:g}')
        if 'pr' in metrics:
            precision, recall = calculate_precision_recall_from_activations(feats_ref, feats_sample)
            print(f'Precision: {precision:g}')
            print(f'Recall: {recall:g}')
    torch.distributed.barrier()

#----------------------------------------------------------------------------

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)

def activations(dataset_path, dest_path, batch):
    """Calculate dataset reference activations."""
    
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    features, s_features, indices, data_num = calculate_activations(image_path=dataset_path, max_batch_size=batch)
    
    temp_dir = f'_buffer_activations'
    os.makedirs(temp_dir, exist_ok=True)
    
    feat_shape = features.shape[1:]
    s_feat_shape = s_features.shape[1:]
    
    np.savez(os.path.join(temp_dir, f'rank_{dist.get_rank()}.npz'), feat=features, s_feat=s_features, ind=indices)
    del features, s_features, indices
    torch.distributed.barrier()
    
    dist.print0(f'Saving dataset activations to "{dest_path}"...')
    if dist.get_rank() == 0:
        # combine results from ranks
        features = np.zeros([data_num, *feat_shape], dtype=np.float64)
        s_features = np.zeros([data_num, *s_feat_shape], dtype=np.float64)
        
        world_size = dist.get_world_size()
        for i in range(world_size):
            data_path = os.path.join(temp_dir, f'rank_{i}.npz')
            ref_rank = dict(np.load(data_path))
            features[ref_rank['ind']] = ref_rank['feat']
            s_features[ref_rank['ind']] = ref_rank['s_feat']
            os.system(f'rm {data_path}')
        
        os.system(f'rm -r {temp_dir}')
        np.savez(dest_path, feat=features, s_feat=s_features)
            
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
