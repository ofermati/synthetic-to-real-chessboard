# train_pix2pix.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from models.networks import NetConfig, build_generator, build_discriminator, GANLoss, init_weights


# -------------------------
# Dataset
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


class PairedFolderDataset(Dataset):
    """
    Expects:
      root/
        A/   (inputs)
        B/   (targets)
    And filenames match between A and B.
    """
    def __init__(self, root: Path, size: int = 256):
        self.root = root
        self.A_dir = root / "A"
        self.B_dir = root / "B"
        assert self.A_dir.is_dir(), f"Missing folder: {self.A_dir}"
        assert self.B_dir.is_dir(), f"Missing folder: {self.B_dir}"

        self.A_paths = list_images(self.A_dir)
        assert len(self.A_paths) > 0, f"No images found in: {self.A_dir}"

        self.size = size
        self.tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.A_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        a_path = self.A_paths[idx]
        b_path = self.B_dir / a_path.name
        if not b_path.exists():
            raise FileNotFoundError(f"Missing matching target for {a_path.name} in {self.B_dir}")

        a = Image.open(a_path).convert("RGB")
        b = Image.open(b_path).convert("RGB")
        x = self.tfm(a)
        y = self.tfm(b)
        return x, y, a_path.name


def load_single_pair(a_path: Path, b_path: Path, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    a = tfm(Image.open(a_path).convert("RGB")).unsqueeze(0)
    b = tfm(Image.open(b_path).convert("RGB")).unsqueeze(0)
    return a, b


# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def save_triplet(x: torch.Tensor, y_fake: torch.Tensor, y: torch.Tensor, out_path: Path) -> None:
    """
    x, y_fake, y: [B,3,H,W] in [-1,1]
    Saves a grid: input | output | target
    """
    # Denorm to [0,1]
    def denorm(t): return (t * 0.5 + 0.5).clamp(0, 1)

    # Use first sample
    x0 = denorm(x[0])
    yf0 = denorm(y_fake[0])
    y0 = denorm(y[0])

    grid = make_grid(torch.stack([x0, yf0, y0], dim=0), nrow=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Train
# -------------------------
def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print("cuda:", torch.cuda.get_device_name(0))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Pix2Pix typical: batch norms + BCE loss
    cfg = NetConfig(img_channels=3, norm_g=args.norm_g, norm_d=args.norm_d, gan_mode=args.gan_mode)
    G = build_generator("unet", cfg, num_downs=args.unet_downs).to(device)
    D = build_discriminator(6, cfg).to(device)  # concat(x, y) => 6 channels

    init_weights(G)
    init_weights(D)

    gan_loss = GANLoss(cfg.gan_mode).to(device)
    l1_loss = nn.L1Loss()

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Data
    if args.single:
        assert args.a_img and args.b_img, "For --single you must pass --a_img and --b_img"
        a_img = Path(args.a_img)
        b_img = Path(args.b_img)
        x_single, y_single = load_single_pair(a_img, b_img, args.size)
        x_single = x_single.to(device)
        y_single = y_single.to(device)