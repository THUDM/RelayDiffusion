## Relay Diffusion: Unifying diffusion process across resolutions for image synthesis <br><sub>Official Pytorch Implementation</sub>

![](resources/samples.jpg)

We propose ***Relay Diffusion Model (RDM)*** as a better framework for diffusion generation. ***RDM*** transfers a low-resolution image or noise into an equivalent high-resolution one via blurring diffusion and block noise. Therefore, the diffusion process can continue seamlessly in any new resolution or model without restarting from pure noise or low-resolution conditioning.

For a formal introduction, Read our paper: [Relay Diffusion: Unifying diffusion process across resolutions for image synthesis](https://github.com/THUDM/RelayDiffusion/blob/main/resources/RelayDiffusion.pdf).

## Setup

### Environment

Download the repo and setup the environment with:

```bash
git clone https://github.com/THUDM/RelayDiffusion.git
cd RelayDiffusion
conda env create -f environment.yml
conda activate rdm
```

We enable `xformers.ops.memory_efficient_attention` to reduce about 15% training cost. If there is no need you can also remove `xformers` from `environment.yml`.

Linux servers with Nvidia A100s are recommended. However, by setting smaller `--batch-gpu` (batch size on a single gpu), you can still run the inference and training scripts on less powerful GPUs.

### Dataset

We preprocess and implement datasets with the same format as [EDM](https://github.com/NVlabs/edm). For CelebA-HQ, follow [*Progressive Growing of GANs for Improved Quality, Stability, and Variation*](https://github.com/tkarras/progressive_growing_of_gans) to construct the high-quality subset of CelebA. For ImageNet, download data from the [official site](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description).

To convert the original data to organized data ready for training at $64\times 64$ or $256\times 256$ resolution, run command:

```bash
python dataset_tool.py \
	--source=/path/to/original/data \
	--dest=/path/to/output/data.zip \
    --transform=center-crop \
	--resolution=64x64 # or --resolution=256x256
```

## Inference & Evaluation

### Sample Generation

To generate samples from RDM models, run command:

```bash
torchrun --standalone --nproc_per_node=1 generate.py --sampler_stages=both --outdir=/path/to/output/dir/ \
    --network_first=/path/to/1st/ckpt --network_second=/path/to/2nd/ckpt
```

To generate $N$ images, set `--seed=[K]-[K+N-1]` with a randomly-picked $K$. You can assign `--nproc_per_node=N` to enable parallel generation of multiple GPUs.

If you want to generate final samples from first-stage results (only use the second stage model), set `--sampler_stages=second` and assign input directory of first-stage results by `--indir`.

Besides, arguments for configurations of the first stage are:

- `num_steps_first`: number of sampling steps.
- `sigma_min_first` & `sigma_max_first`: lowest & highest noise level.
- `rho_first`: time step exponent.
- `cfg_scale_first`: scale of classifier-free guidance.
- `S_churn`: stochasticity strength.
- `S_min` & `S_max`: min & max noise level.
- `S_noise`: noise inflation.

Arguments for configurations of the second stage are:

- `num_steps_second`: number of sampling steps.
- `sigma_min_second` & `sigma_max_second`: lowest & highest noise level.
- `blur_sigma_max_second`: maximum sigma of blurring schedule.
- `rho_second`: time step exponent.
- `cfg_scale_second`: scale of classifier-free guidance.
- `up_scale_second`: scale of upsampling.
- `truncation_sigma_second` & `truncation_t_second`: truncation point of noise & time schedule.
- `s_block_second`: strength of block noise addition.
- `s_noise_second`: strength of stochasticity.


### Evaluation Metrics

We quantitatively measure the sample quality by metrics including **Fréchet inception distance (FID)**, **spatial FID (sFID)**, **Inception Score (IS)**, **Precision** and **Recall**. For sFID, IS, Precision and Recall, we reformat the calculation pipeline based on the formulation in `tensorflow` from [ADM](https://github.com/openai/guided-diffusion).

First, run the following command to generate activation data file from samples and dataset:

```bash
torchrun --standalone --nproc_per_node=1 evaluate.py activations --data=/sample/dir/ --dest=eval-refs/activations_sample.npz --batch=64 # build sample activations
torchrun --standalone --nproc_per_node=1 evaluate.py activations --data=/path/to/dataset.zip --dest=eval-refs/activations_ref.npz --batch=64 # build reference activations
```

Then calculate metrics based on pre-built activations, run command:

```bash
torchrun --standalone --nproc_per_node=1 evaluate.py calc --batch=64 \
    --activations_sample=eval-refs/activations_sample.npz \
    --activations_ref=eval-refs/activations_ref.npz \
    [-m fid] [-m sfid] [-m is] [-m pr] \ # assign metrics to be calculated
```

### Performance Reproduction

RDM achieves competitive results in comparison with previous SoTA models:

| Dataset   | Resolution | Training Samples | FID  | sFID |   IS   | Precision | Recall |
| --------- | ---------- | ---------------- | :--: | :--: | :----: | :-------: | :----: |
| CelebA-HQ | 256x256    | 47M              | 3.15 |  -   |   -    |   0.77    |  0.55  |
| ImageNet  | 256x256    | 1250M            | 1.87 | 3.97 | 278.75 |   0.81    |  0.59  |

We provide best pre-trained checkpoints of RDM and their sampler settings for reproducing performance:

- CelebA-HQ $256\times 256$:

  Download checkpoints of [first stage](https://cloud.tsinghua.edu.cn/f/8e8e4b2743fe4447b497/?dl=1) and [second stage](https://cloud.tsinghua.edu.cn/f/b8cd559a0e9f4b9abd39/?dl=1), place them in `ckpts/`, generate samples and their activations by commands:

  ```bash
  torchrun --standalone --nproc_per_node=8 generate_celebahq.py --outdir=generations/celebahq_samples/ \
      --network_first=ckpts/celebahq_first_stage.pt \
      --network_second=ckpts/celebahq_second_stage.pt
  torchrun --standalone --nproc_per_node=1 evaluate.py activations \
      --data=generations/celebahq_samples/ --dest=eval-refs/celebahq_act_sample.npz 
  ```

  Generate activation data from CelebA-HQ zip or download our version from [here](https://cloud.tsinghua.edu.cn/f/a26f714e36304c3e948d/?dl=1):

  ```bash
  torchrun --standalone --nproc_per_node=1 evaluate.py activations \
      --data=datasets/celebahq-256x256.zip --dest=eval-refs/celebahq_act_ref.npz 
  ```

  Calculate metrics by command:

  ```bash
  python evaluate.py calc -m fid -m pr \
      --activations_sample=eval-refs/celebahq_act_sample.npz \
      --activations_ref=eval-refs/celebahq_act_ref.npz
  ```

- ImageNet $256\times 256$:

  Download checkpoints of [first stage](https://cloud.tsinghua.edu.cn/f/c9a0ab6341704ed0be55/?dl=1) and [second stage](https://cloud.tsinghua.edu.cn/f/b5915d0b7d994e86b4bb/?dl=1), place them in `ckpts/`, generate samples and their activations by commands:

  ```bash
  torchrun --standalone --nproc_per_node=8 generate_imagenet.py --outdir=generations/imagenet_samples/ \
      --network_first=ckpts/imagenet_first_stage.pkl \
      --network_second=ckpts/imagenet_second_stage.pt
  torchrun --standalone --nproc_per_node=1 evaluate.py activations \
      --data=generations/imagenet_samples/ --dest=eval-refs/imagenet_act_sample.npz 
  ```

  Generate activation data from ImageNet zip:

  ```bash
  torchrun --standalone --nproc_per_node=1 evaluate.py activations \
      --data=datasets/imagenet-256x256.zip --dest=eval-refs/imagenet_act_ref.npz 
  ```

  Calculate FID, sFID and IS by command:

  ```bash
  python evaluate.py calc -m fid -m sfid -m is \
      --activations_sample=eval-refs/imagenet_act_sample.npz \
      --activations_ref=eval-refs/imagenet_act_ref.npz
  ```

  For the calculation of Precision and Recall on ImageNet, we follow [ADM](https://github.com/openai/guided-diffusion) to use 1w reference samples. You can download the activation data we produced from [here](https://cloud.tsinghua.edu.cn/f/924f9878ddc340bcb09c/?dl=1). Then run the following command:

  ```bash
  python evaluate.py calc -m pr \
      --activations_sample=eval-refs/imagenet_act_sample.npz \
      --activations_ref=eval-refs/imagenet_act_1w_ref.npz
  ```

## Training

you can follow the instruction of [EDM](https://github.com/NVlabs/edm) to train a new model of the first stage (standard diffusion). Using ImageNet for example, run command:

```bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/imagenet-64x64.zip --eff-attn=1 \
	--cond=1 --batch=4096  --batch-gpu=64 --lr=1e-4 --ema=50 --dropout=0.1 --fp16=1 --ls=25
```

If you want to train a second stage model (blurring diffusion), set argument `--precond=blur` and other arguments for the configuration of blurring diffusion. The command will be:

```bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/imagenet-256x256.zip --eff-attn=1 \
	--cond=1 --batch=4096  --batch-gpu=32 --lr=1e-4 --dropout=0.1 --fp16=1 --ls=1 \
	--precond=blur --up-scale=4 --block-scale=0.15 --prob-length=0.93 --blur-sigma-max=3
```

As for CelebA-HQ, train a first stage model with:

```bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/CelebA-HQ-64x64.zip --eff-attn=1 \
	--cond=1 --batch=4096  --batch-gpu=64 --lr=1e-4 --ema=50 --dropout=0.1 --ls=25
```

And for training a second stage model:

```bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/CelebA-HQ-256x256.zip --eff-attn=1 \
	--cond=0 --batch=1024  --batch-gpu=8 --lr=1e-4 --dropout=0.2 --augment=0.2 --fp16=1 --ls=1 \
	--arch=adm --precond=blur --up-scale=4 --block-scale=0.15 --prob-length=0.93 --blur-sigma-max=2.0
```

## Citation

## Acknowledgements

This implementation is based on https://github.com/NVlabs/edm (codebase of EDM). Thanks a lot!