#  Denoising Diffusion Probabilistic Models (DDPM) - My implementation with UNet backbone
This repository includes **DDPM implementation** , **Training pipeline on CelebA Dataset** & **Use of EMA (Exponential Moving Average) in generating new samples**
---

## Model
### DDPM with UNet Backbone:
- Inspired by Vanilla Unet : [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597) (combined with MultiHeadAttention blocks)
- Implemented to generate (32x32 RGB images or 64x64 RGB images) (However there is a space for improvement to upgrade model to work better on 64x64 or larger images)
- Please check `unet.py` to see full details
---
## Dataset:
- **CelebA** - dataset was selected due to its rich annotation of facial attributes (e.g., smiling, wearing glasses), which is crucial for experiments involving the manipulation and extraction of these features within the VAE's latent space [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
---

##  Goal:
- To understand Diffusion Models fundamentals and generate new face samples acording to CelebA dataset

## Training Details
All training details you can find in `ddpm_test.ipynb` notebook including training process, hyperparemeters etc. Training loop is available in `train.py`

Note: Due to task complexity training in the notebook is not completed but you can use the code from the notebook to start the training.

## Space for improvement & experiments
- Expanding model architecture by adding one piece of `ResidualBlock` combined with `MultiHeadAttentionBlock` and `Downsample` (and `Upsample` on the other side of course) 
- Training on smaller part of **CelebA** (for example only for males/females)

