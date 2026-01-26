# train_pix2pix.py
import argparse
import os
import sys
import random
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

from models.networks import NetConfig, build_generator, build_discriminator, GANLoss, init_weights


# ======================
# CONFIG & ARGS
# ======================
IMG_SIZE = 256
LOAD_SIZE = 286 # Load larger for random crop
BATCH_SIZE = 1
EPOCHS = 200
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_L1 = 100.0  # Weight for L1 loss

DATA_ROOT = Path("datasets/paired")
SYN_DIR = DATA_ROOT / "synthetic"
REAL_DIR = DATA_ROOT / "real"
OUT_DIR = Path("outputs/pix2pix_run1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "training_log.csv"

# -------------------------
# Utils
# -------------------------
def init_logger():
    """Initialize CSV logger with headers if file doesn't exist"""
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", 
                "Loss_G_Total", 
                "Loss_G_GAN",
                "Loss_G_L1",
                "Loss_D"
            ])
        print(f"📄 Created new log file: {LOG_FILE}")
    else:
        print(f"📄 Appending to existing log file: {LOG_FILE}")

def log_losses(epoch, losses):
    """Append a row of losses to CSV"""
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, 
            f"{losses['G_Total']:.4f}",
            f"{losses['G_GAN']:.4f}",
            f"{losses['G_L1']:.4f}",
            f"{losses['D']:.4f}"
        ])
# -------------------------
class PairedTransform:
    """
    Applies consistent random transforms (Crop, Flip) to both A and B images.
    """
    def __init__(self, load_size=286, fine_size=256):
        self.load_size = load_size
        self.fine_size = fine_size

    def __call__(self, img_a, img_b):
        # 1. Resize to slightly larger
        img_a = TF.resize(img_a, (self.load_size, self.load_size), interpolation=Image.BICUBIC)
        img_b = TF.resize(img_b, (self.load_size, self.load_size), interpolation=Image.BICUBIC)

        # 2. Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=(self.fine_size, self.fine_size))
        img_a = TF.crop(img_a, i, j, h, w)
        img_b = TF.crop(img_b, i, j, h, w)

        # 3. Random Horizontal Flip
        if random.random() > 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)

        # 4. ToTensor and Normalize
        img_a = TF.to_tensor(img_a)
        img_b = TF.to_tensor(img_b)
        
        # Normalize to [-1, 1]
        img_a = TF.normalize(img_a, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img_b = TF.normalize(img_b, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        return img_a, img_b


# -------------------------
# Dataset
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

class PairedRecursiveDataset(Dataset):
    """
    Recursively finds matching images in two directory trees.
    Assumes relative paths match.
    Example:
      root_A/Game2/frame_001.jpg
      root_B/Game2/frame_001.jpg
    """
    def __init__(self, root_A: Path, root_B: Path, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        
        # Find all images in root_A recursively
        self.A_paths = []
        self.B_paths = []
        
        # Walk through root_A
        for path in sorted(root_A.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                # Construct expected path in root_B
                rel_path = path.relative_to(root_A)
                target_path = root_B / rel_path
                
                # Check if it exists in root_B
                if target_path.exists():
                    self.A_paths.append(path)
                    self.B_paths.append(target_path)
                
        print(f"Found {len(self.A_paths)} paired images.")

    def __len__(self) -> int:
        return len(self.A_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a_path = self.A_paths[idx]
        b_path = self.B_paths[idx]

        # A = Input (Synthetic), B = Target (Real)
        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")

        if self.transform:
            a_tensor, b_tensor = self.transform(a_img, b_img)
        else:
            # Fallback (should not happen with our setup)
            t = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*3, (0.5,)*3)
            ])
            a_tensor = t(a_img)
            b_tensor = t(b_img)
            
        return a_tensor, b_tensor


# -------------------------
# Helpers
# -------------------------
def save_sample_images(real_A, fake_B, real_B, epoch, save_dir):
    """
    Saves a grid: Synthetic Input | Generated Real | Ground Truth Real
    """
    # Stack images width-wise
    grid = torch.cat((real_A, fake_B, real_B), dim=3)
    save_path = save_dir / f"sample_epoch_{epoch}.png"
    save_image(grid, save_path, normalize=True)


# -------------------------
# Train Loop
# -------------------------
def train():
    print(f"Device: {DEVICE}")
    
    # 1. Dataset & Transforms
    paired_transform = PairedTransform(load_size=LOAD_SIZE, fine_size=IMG_SIZE)
    dataset = PairedRecursiveDataset(SYN_DIR, REAL_DIR, transform=paired_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    if len(dataset) == 0:
        print("No paired images found! Check paths.")
        return

    # 2. Models
    # NetConfig: norm_g='batch' is standard for Pix2Pix
    cfg = NetConfig(norm_g="batch", norm_d="batch", gan_mode="lsgan")
    
    # Generator: UNet is standard for Pix2Pix
    G = build_generator("unet", cfg).to(DEVICE)
    # Discriminator: PatchGAN
    # Pix2Pix Discriminator takes concatenation of (Input, Real/Fake) -> 6 channels
    D = build_discriminator(6, cfg).to(DEVICE) 

    init_weights(G)
    init_weights(D)

    # 4. Losses
    gan_loss = GANLoss("lsgan").to(DEVICE)
    l1_loss = nn.L1Loss()

    # 5. Resume Logic
    start_epoch = 1
    # Check for existing checkpoints
    checkpoints = list(OUT_DIR.glob("G_epoch*.pth"))
    if len(checkpoints) > 0:
        try:
            epochs = []
            for p in checkpoints:
                try:
                    ep_num = int(p.stem.split("epoch")[-1])
                    epochs.append(ep_num)
                except ValueError:
                    continue
            
            if len(epochs) > 0:
                last_epoch = max(epochs)
                print(f"\nFound checkpoint for epoch {last_epoch}. Resuming training from epoch {last_epoch+1}...")
                
                # Load Generator
                path_g = OUT_DIR / f"G_epoch{last_epoch}.pth"
                if path_g.exists():
                    G.load_state_dict(torch.load(path_g, map_location=DEVICE))
                    print("Generator weights loaded.")
                
                # Try to load Discriminator (if saved)
                path_d = OUT_DIR / f"D_epoch{last_epoch}.pth"
                if path_d.exists():
                    D.load_state_dict(torch.load(path_d, map_location=DEVICE))
                    print("Discriminator weights loaded.")
                else:
                    print("Warning: Discriminator weights not found. It will be initialized from scratch.")
                
                start_epoch = last_epoch + 1
        except Exception as e:
            print(f"Error trying to resume: {e}. Starting from scratch.")

    # Initialize Logger
    if start_epoch == 1:
        # Create new log file only if starting from scratch
        if LOG_FILE.exists():
             print(f"⚠️  Log file {LOG_FILE} exists. Cleaning up for new run...")
             LOG_FILE.unlink()
        init_logger()
    else:
        print(f"📄 Appending to existing log file: {LOG_FILE}")
        # Ensure header exists if file is missing but we are resuming
        if not LOG_FILE.exists():
            init_logger()

    # 3. Optimizers (Initialize after loading weights if we wanted to load optimizer state too, but here we just init fresh)
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # 6. Loop
    for epoch in range(start_epoch, EPOCHS + 1):
        loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        # Accumulators for epoch logging
        running_losses = {
            'G_Total': 0.0,
            'G_GAN': 0.0,
            'G_L1': 0.0,
            'D': 0.0
        }
        steps = 0
        
        for i, (real_A, real_B) in enumerate(loop):
            # real_A = Synthetic (Input)
            # real_B = Real (Target)
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            # ------------------
            # Train Discriminator
            # ------------------
            opt_D.zero_grad()

            # Fake
            fake_B = G(real_A)
            # Detach fake_B to avoid backprop to G
            fake_AB = torch.cat((real_A, fake_B.detach()), 1)
            pred_fake = D(fake_AB)
            loss_D_fake = gan_loss(pred_fake, False)

            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = D(real_AB)
            loss_D_real = gan_loss(pred_real, True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_D.step()

            # ------------------
            # Train Generator
            # ------------------
            opt_G.zero_grad()
            
            # GAN Loss (G tries to fool D)
            # Standard practice: Do forward once, but be careful with detach.
            # Easier/Cleaner: Re-forward for G step to keep graph clean.
            fake_B = G(real_A)
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = D(fake_AB)
            
            loss_G_GAN = gan_loss(pred_fake, True)
            
            # L1 Loss (Pixel-wise reconstruction)
            loss_G_L1 = l1_loss(fake_B, real_B) * LAMBDA_L1
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()
            
            # Update running losses
            steps += 1
            running_losses['G_Total'] += loss_G.item()
            running_losses['G_GAN'] += loss_G_GAN.item()
            running_losses['G_L1'] += loss_G_L1.item()
            running_losses['D'] += loss_D.item()

            if i % 50 == 0:
                loop.set_postfix(G_GAN=loss_G_GAN.item(), G_L1=loss_G_L1.item(), D=loss_D.item())

        # ======================
        # End of Epoch
        # ======================
        
        # Calculate averages
        avg_losses = {k: v / steps for k, v in running_losses.items()}
        
        # Log to CSV
        log_losses(epoch, avg_losses)

        torch.save(G.state_dict(), OUT_DIR / f"G_epoch{epoch}.pth")
        torch.save(D.state_dict(), OUT_DIR / f"D_epoch{epoch}.pth") # Save Discriminator too for proper resume
        
        with torch.no_grad():
            # Save sample from last batch
            save_sample_images(real_A, fake_B, real_B, epoch, OUT_DIR)

    print("Training finished.")


if __name__ == "__main__":
    train()