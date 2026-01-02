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
        self.real = list(real_dir.rglob("*.png"))
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

G_A2B = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)
G_B2A = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)

D_A = build_discriminator(3, cfg).to(DEVICE)
D_B = build_discriminator(3, cfg).to(DEVICE)

init_weights(G_A2B)
init_weights(G_B2A)
init_weights(D_A)
init_weights(D_B)

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

opt_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

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

        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        loss_GAN_A2B = gan_loss(D_B(fake_B), True)
        loss_GAN_B2A = gan_loss(D_A(fake_A), True)

        rec_A = G_B2A(fake_B)
        rec_B = G_A2B(fake_A)

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
        # Train D_B
        # ------------------
        opt_D_B.zero_grad()
        loss_D_B = (
            gan_loss(D_B(real_B), True) +
            gan_loss(D_B(fake_B.detach()), False)
        ) * 0.5
        loss_D_B.backward()
        opt_D_B.step()

    # ======================
    # Save checkpoints
    # ======================
    torch.save(G_A2B.state_dict(), OUT_DIR / f"G_A2B_epoch{epoch}.pth")
    torch.save(G_B2A.state_dict(), OUT_DIR / f"G_B2A_epoch{epoch}.pth")

print("✅ Training finished")
