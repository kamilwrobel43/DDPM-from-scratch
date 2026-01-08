import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusion import Diffusion
from ema import EMA
import os



def plot_images(imgs):
    imgs = imgs.cpu()
    plt.figure(figsize=(12, 4))
    
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        
        img_permuted = img.permute(1, 2, 0)
    
        if not img_permuted.dtype == torch.uint8:
            img_permuted = (img_permuted.clamp(-1, 1) + 1) / 2
            img_to_show = img_permuted.numpy()
        else:
            img_to_show = img_permuted.numpy()
            
        plt.imshow(img_to_show)
        plt.axis('off') 
    
    plt.show()

scaler = torch.GradScaler()

def train_epoch(model, diffusion: Diffusion, optimizer, train_loader, loss_fn, scheduler, ema: EMA | None = None):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc = "Training")
    for (img, _) in pbar:

        img = img.to(diffusion.device)
        t = diffusion.sample_t(img.shape[0])
        
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            noised_img, eps = diffusion.noise_images(img, t)
            predicted_eps = model(noised_img,t)
            loss = loss_fn(eps, predicted_eps)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_train_loss += loss.item()
        if ema:
            ema.update()

    total_train_loss /= len(train_loader)
    return total_train_loss


def eval_epoch(model, diffusion: Diffusion, optimizer, test_loader, loss_fn):
    model.eval()
    total_eval_loss = 0.0
    pbar = tqdm(test_loader, desc = "Evaluating")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for (img, _) in pbar:

                img = img.to(diffusion.device)
                t = diffusion.sample_t(img.shape[0])
                noised_img, eps = diffusion.noise_images(img, t)
                predicted_eps = model(noised_img,t)

                loss = loss_fn(eps, predicted_eps)
                total_eval_loss += loss.item()

    total_eval_loss /= len(test_loader)
    return total_eval_loss
        

def train_model(model, diffusion: Diffusion, optimizer, train_loader, test_loader, loss_fn, scheduler, n_epochs, ema: EMA | None):

    os.makedirs("checkpoints", exist_ok=True)
    model = torch.compile(model)

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, diffusion, optimizer, train_loader, loss_fn, scheduler, ema)
        eval_loss = eval_epoch(model, diffusion, optimizer, test_loader, loss_fn)
        print(f"Epoch: {epoch+1}/{n_epochs} train loss: {train_loss:.4f} | eval loss: {eval_loss:.4f}")

        if (epoch+1) % 10 == 0:
            if ema:
                ema.apply_shadow()

            imgs = diffusion.sample_img(model, 5)
            plot_images(imgs)

            if ema:
                ema.restore()

                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema': ema.shadow
                }
            else:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

            torch.save(checkpoint, f"checkpoints/weights32_{epoch+1}_epoch.pth")