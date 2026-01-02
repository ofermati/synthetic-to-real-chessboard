import itertools
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.networks import (
    NetConfig,
    build_generator,
    build_discriminator,
    GANLoss,
    init_weights,
)

# ======================
# CONFIG
# ======================
DATA_ROOT = Path("datasets/unpaired")
SYN_DIR = DATA_ROOT / "synthetic"   # synthetic
REAL_DIR = DATA_ROOT / "real"   # real

OUT_DIR = Path("outputs/cyclegan_run1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 50
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Dataset
# ======================
class UnpairedImageDataset(Dataset):
    def __init__(self, synthetic_dir, real_dir, transform):
        self.synthetic = list(synthetic_dir.rglob("*.png"))
        self.real = list(real_dir.rglob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return max(len(self.synthetic), len(self.real))

    def __getitem__(self, idx):
        img_syn = Image.open(self.synthetic[idx % len(self.synthetic)]).convert("RGB")
        img_real = Image.open(self.real[idx % len(self.real)]).convert("RGB")
        return self.transform(img_syn), self.transform(img_real)

# ======================
# Transforms
# ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

dataset = UnpairedImageDataset(SYN_DIR, REAL_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# Models
# ======================
cfg = NetConfig()

G_S2R = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)
G_R2S = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)

D_S = build_discriminator(3, cfg).to(DEVICE)
D_R = build_discriminator(3, cfg).to(DEVICE)
init_weights(G_S2R)
init_weights(G_R2S)
init_weights(D_S)
init_weights(D_R)
# ======================
# Losses
# ======================
gan_loss = GANLoss("lsgan").to(DEVICE)
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()

# ======================
# Optimizers
# ======================
opt_G = optim.Adam(
    itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
    lr=LR, betas=(0.5, 0.999)
)

opt_D_S = optim.Adam(D_S.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D_R = optim.Adam(D_R.parameters(), lr=LR, betas=(0.5, 0.999))

# ======================
# Training loop
# ======================
for epoch in range(1, EPOCHS + 1):
    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for real_A, real_B in loop:
        real_A = real_A.to(DEVICE)
        real_B = real_B.to(DEVICE)

        # ------------------
        # Train Generators
        # ------------------
        opt_G.zero_grad()

        fake_B = G_S2R(real_A)
        fake_A = G_R2S(real_B)

        loss_GAN_S2R = gan_loss(D_R(fake_B), True)
        loss_GAN_R2S = gan_loss(D_S(fake_A), True)

        rec_A = G_R2S(fake_B)
        rec_B = G_S2R(fake_A)
        loss_cycle = cycle_loss(rec_A, real_A) + cycle_loss(rec_B, real_B)

        loss_G = loss_GAN_A2B + loss_GAN_B2A + 10 * loss_cycle
        loss_G.backward()
        opt_G.step()

        # ------------------
        # Train D_A
        # ------------------
        opt_D_A.zero_grad()
        loss_D_A = (
            gan_loss(D_A(real_A), True) +
            gan_loss(D_A(fake_A.detach()), False)
        ) * 0.5
        loss_D_A.backward()
        opt_D_A.step()

        # ------------------
        # Train D_R
        # ------------------
        opt_D_R.zero_grad()
        loss_D_R = (
            gan_loss(D_R(real_B), True) +
            gan_loss(D_R(fake_B.detach()), False)
        ) * 0.5
        loss_D_R.backward()
        opt_D_R.step()

    # ======================
    # Save checkpoints
    # ======================
    torch.save(G_S2R.state_dict(), OUT_DIR / f"G_S2R_epoch{epoch}.pth")
    torch.save(G_R2S.state_dict(), OUT_DIR / f"G_R2S_epoch{epoch}.pth")

print("Training finished")
