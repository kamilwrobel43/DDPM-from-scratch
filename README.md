# PyTorch Implementation of Denoising Diffusion Probabilistic Models (DDPM)

This repository contains a clean, from-scratch implementation of **DDPM** with a custom **UNet backbone**, designed for generating high-quality face samples using the **CelebA dataset**.

The project focuses on understanding the fundamentals of diffusion models, implementing stable training pipelines, and utilizing advanced techniques like **Exponential Moving Average (EMA)** for better generation quality.

---

## ⚡ Key Features
* **Custom UNet Architecture**: Inspired by the original U-Net, enhanced with **Multi-Head Attention** mechanisms and **Residual Blocks**.
* **EMA (Exponential Moving Average)**: Implemented shadow weights to stabilize image generation and improve visual fidelity.
* **Mixed Precision Training**: Utilizes `torch.autocast` (fp16) for faster training and lower memory usage.
* **Gradient Clipping**: Added to prevent gradient explosion and ensure stable convergence.
* **Configurable Pipeline**: Supports training on 32x32 and 64x64 resolutions.

---

## 🧠 Model Architecture

### DDPM with UNet Backbone
The model is built upon the foundational ideas of Vanilla U-Net [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597), adapted for the diffusion process:
* **Sinusoidal Positional Embeddings**: To inform the model about the current timestep $t$.
* **Residual Blocks**: For deep feature extraction without vanishing gradients.
* **Multi-Head Attention**: Applied at lower resolutions (e.g., 16x16, 8x8) to capture global dependencies.
* **Bottleneck**: A robust middle section combining ResBlocks and Attention.

Full architectural details can be found in `unet.py`.

---

## 📂 Dataset

The model is trained on the **CelebA Dataset** (CelebFaces Attributes Dataset).
* **Source**: [Kaggle - CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
* **Preprocessing**: Images are resized and normalized to [-1, 1] range.

---

## 🚀 Training Details

### Structure
* `train.py`: Contains the main training loop, including optimization steps, EMA updates, and checkpoint saving.
* `ddpm_test.ipynb`: A Jupyter Notebook demonstrating the training process, hyperparameter configuration, and visualization of results.
* `utills.py`: Contains a function to load checkpoint to renew training process
* `ema.py`: Contains universal EMA class
* `duffusion.py`: Contains Diffusion class that includes noising and sampling functions

### Note on Training
The training pipeline in the notebook is set up for demonstration. For full convergence (especially on 64x64 images), it is recommended to run the training for **150-200 epochs** using a GPU.
Feel free to experiment with hyperparameters to get more accurate results

---

## 🔮 Roadmap & Future Improvements

To further improve the model's performance and capability, the following experiments are planned:
* **Architecture Expansion**: Adding an additional downsampling stage (with corresponding ResBlocks and Attention) to better handle fine details in 64x64 or 128x128 resolution.
* **Dataset Filtering**: Training on specific subsets of CelebA (e.g., filtered by gender or specific attributes) to test conditional generation capabilities.
* **Scheduler Optimization**: Experimenting with `CosineAnnealingLR` to improve convergence in later stages of training.
