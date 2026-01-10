import itertools
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision.utils import save_image

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "cut_8X8"
SYN_DIR = DATA_ROOT / "synthetic"
REAL_DIR = DATA_ROOT / "real"

OUT_DIR = Path("outputs/cyclegan_run1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 50
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESUME = True
RESUME_EPOCH = 8  # ימשיך מ-epoch 9

# ======================
# Dataset
# ======================
class UnpairedImageDataset(Dataset):
    def __init__(self, synthetic_dir, real_dir, transform):
        self.synthetic = sorted(list(synthetic_dir.rglob("*.png")))
        self.real      = sorted(list(real_dir.rglob("*.png")))
        assert len(self.synthetic) > 0
        assert len(self.real) > 0
        self.transform = transform

    def __len__(self):
        return min(len(self.synthetic), len(self.real))

    def __getitem__(self, idx):
        img_syn = Image.open(self.synthetic[idx % len(self.synthetic)]).convert("RGB")
        img_real = Image.open(self.real[idx % len(self.real)]).convert("RGB")
        return self.transform(img_syn), self.transform(img_real)

# ======================
# Transforms
# ======================
transform = transforms.Compose([
    transforms.Resize((152, 152)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

dataset = UnpairedImageDataset(SYN_DIR, REAL_DIR, transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
# ======================
# Models
# ======================
cfg = NetConfig()

G_S2R = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)
G_R2S = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)
D_S = build_discriminator(3, cfg).to(DEVICE)
D_R = build_discriminator(3, cfg).to(DEVICE)

# אם לא עושים resume - מאתחלים משקולות
if not RESUME:
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
    itertools.chain(G_S2R.parameters(), G_R2S.parameters()),
    lr=LR, betas=(0.5, 0.999)
)
opt_D_S = optim.Adam(D_S.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D_R = optim.Adam(D_R.parameters(), lr=LR, betas=(0.5, 0.999))

# ======================
# Resume (load weights)
# ======================
def load_epoch_weights(epoch: int) -> None:
    G_S2R.load_state_dict(torch.load(OUT_DIR / f"G_S2R_epoch{epoch}.pth", map_location=DEVICE))
    G_R2S.load_state_dict(torch.load(OUT_DIR / f"G_R2S_epoch{epoch}.pth", map_location=DEVICE))
    D_S.load_state_dict(torch.load(OUT_DIR / f"D_S_epoch{epoch}.pth", map_location=DEVICE))
    D_R.load_state_dict(torch.load(OUT_DIR / f"D_R_epoch{epoch}.pth", map_location=DEVICE))

if RESUME:
    # בדיקת קבצים כדי שלא ניפול באמצע
    needed = [
        OUT_DIR / f"G_S2R_epoch{RESUME_EPOCH}.pth",
        OUT_DIR / f"G_R2S_epoch{RESUME_EPOCH}.pth",
        OUT_DIR / f"D_S_epoch{RESUME_EPOCH}.pth",
        OUT_DIR / f"D_R_epoch{RESUME_EPOCH}.pth",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing resume checkpoint files:\n" + "\n".join(str(p) for p in missing))

    load_epoch_weights(RESUME_EPOCH)
    print(f"Resuming from epoch {RESUME_EPOCH} (starting epoch {RESUME_EPOCH + 1})")

start_epoch = RESUME_EPOCH + 1 if RESUME else 1

# ======================
# Samples helpers
# ======================
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

SAMPLES_DIR = OUT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

fixed_sample = None

# ======================
# Training loop
# ======================
for epoch in range(start_epoch, EPOCHS + 1):
    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for real_S, real_R in loop:
        real_S = real_S.to(DEVICE)
        real_R = real_R.to(DEVICE)

        if fixed_sample is None:
            fixed_sample = (real_S[:1].detach().clone(), real_R[:1].detach().clone())

        # ---- Train Generators ----
        opt_G.zero_grad()

        fake_R = G_S2R(real_S)
        fake_S = G_R2S(real_R)

        loss_GAN_S2R = gan_loss(D_R(fake_R), True)
        loss_GAN_R2S = gan_loss(D_S(fake_S), True)

        rec_S = G_R2S(fake_R)
        rec_R = G_S2R(fake_S)

        loss_cycle = cycle_loss(rec_S, real_S) + cycle_loss(rec_R, real_R)
        loss_id = identity_loss(G_S2R(real_R), real_R) + identity_loss(G_R2S(real_S), real_S)

        loss_G = loss_GAN_S2R + loss_GAN_R2S + 10 * loss_cycle + 5 * loss_id
        loss_G.backward()
        opt_G.step()

        # ---- Train D_S ----
        opt_D_S.zero_grad()
        loss_D_S = (gan_loss(D_S(real_S), True) + gan_loss(D_S(fake_S.detach()), False)) * 0.5
        loss_D_S.backward()
        opt_D_S.step()

        # ---- Train D_R ----
        opt_D_R.zero_grad()
        loss_D_R = (gan_loss(D_R(real_R), True) + gan_loss(D_R(fake_R.detach()), False)) * 0.5
        loss_D_R.backward()
        opt_D_R.step()

    # ======================
    # Save sample images
    # ======================
    G_S2R.eval()
    G_R2S.eval()
    with torch.no_grad():
        s0, r0 = fixed_sample
        fake_r0 = G_S2R(s0)
        fake_s0 = G_R2S(r0)
        rec_s0  = G_R2S(fake_r0)
        rec_r0  = G_S2R(fake_s0)

        row1 = torch.cat([denorm(s0), denorm(fake_r0), denorm(rec_s0)], dim=0)
        row2 = torch.cat([denorm(r0), denorm(fake_s0), denorm(rec_r0)], dim=0)

        grid = torchvision.utils.make_grid(torch.cat([row1, row2], dim=0), nrow=3)
        save_image(grid, SAMPLES_DIR / f"epoch_{epoch:03d}.png")

    G_S2R.train()
    G_R2S.train()

    # ======================
    # Save checkpoints (epoch)
    # ======================
    torch.save(G_S2R.state_dict(), OUT_DIR / f"G_S2R_epoch{epoch}.pth")
    torch.save(G_R2S.state_dict(), OUT_DIR / f"G_R2S_epoch{epoch}.pth")
    torch.save(D_S.state_dict(), OUT_DIR / f"D_S_epoch{epoch}.pth")
    torch.save(D_R.state_dict(), OUT_DIR / f"D_R_epoch{epoch}.pth")

print("Training finished")
