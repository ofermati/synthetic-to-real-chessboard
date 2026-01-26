import itertools
import sys
import csv
import time
from datetime import datetime
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
RESUME_EPOCH = 39

# ======================
# Logging dirs/files
# ======================
LOGS_DIR = OUT_DIR / "logsFromTheStart"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOSS_CSV = LOGS_DIR / "losses.csv"
META_TXT = LOGS_DIR / "run_meta.txt"

SAMPLES_DIR = OUT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

FIXED_DIR = OUT_DIR / "fixed_sample"
FIXED_DIR.mkdir(parents=True, exist_ok=True)

BEST_DIR = OUT_DIR / "best"
BEST_DIR.mkdir(parents=True, exist_ok=True)

# אם מתחילים ריצה חדשה (לא resume) - נכתוב CSV מחדש עם כותרות
if not RESUME:
    with open(LOSS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "epoch",
            "loss_G", "loss_D_S", "loss_D_R",
            "loss_GAN_S2R", "loss_GAN_R2S",
            "loss_cycle", "loss_id"
        ])

# metadata למצגת
if not META_TXT.exists():
    with open(META_TXT, "w") as f:
        f.write(f"run_started: {datetime.now().isoformat()}\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"lr: {LR}\n")
        f.write(f"img_size: 152x152\n")
        f.write(f"resume: {RESUME}\n")
        f.write(f"resume_epoch: {RESUME_EPOCH}\n")
        f.write(f"syn_dir: {SYN_DIR}\n")
        f.write(f"real_dir: {REAL_DIR}\n")

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
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
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
# Helpers
# ======================
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

fixed_sample = None
best_loss_G = float("inf")

# ======================
# Training loop
# ======================
for epoch in range(start_epoch, EPOCHS + 1):
    epoch_start = time.time()

    sum_loss_G = 0.0
    sum_loss_D_S = 0.0
    sum_loss_D_R = 0.0
    sum_loss_GAN_S2R = 0.0
    sum_loss_GAN_R2S = 0.0
    sum_loss_cycle = 0.0
    sum_loss_id = 0.0
    n_batches = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for real_S, real_R in loop:
        real_S = real_S.to(DEVICE, non_blocking=True)
        real_R = real_R.to(DEVICE, non_blocking=True)

        if fixed_sample is None:
            fixed_sample = (real_S[:1].detach().clone(), real_R[:1].detach().clone())
            # save fixed inputs for presentation
            save_image(denorm(fixed_sample[0].cpu()), FIXED_DIR / "fixed_S.png")
            save_image(denorm(fixed_sample[1].cpu()), FIXED_DIR / "fixed_R.png")

        # ---- Train Generators ----
        opt_G.zero_grad(set_to_none=True)

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
        opt_D_S.zero_grad(set_to_none=True)
        loss_D_S = (gan_loss(D_S(real_S), True) + gan_loss(D_S(fake_S.detach()), False)) * 0.5
        loss_D_S.backward()
        opt_D_S.step()

        # ---- Train D_R ----
        opt_D_R.zero_grad(set_to_none=True)
        loss_D_R = (gan_loss(D_R(real_R), True) + gan_loss(D_R(fake_R.detach()), False)) * 0.5
        loss_D_R.backward()
        opt_D_R.step()

        # ---- Accumulate for epoch averages ----
        n_batches += 1
        sum_loss_G += loss_G.item()
        sum_loss_D_S += loss_D_S.item()
        sum_loss_D_R += loss_D_R.item()
        sum_loss_GAN_S2R += loss_GAN_S2R.item()
        sum_loss_GAN_R2S += loss_GAN_R2S.item()
        sum_loss_cycle += loss_cycle.item()
        sum_loss_id += loss_id.item()

        loop.set_postfix({
            "G": f"{loss_G.item():.3f}",
            "D_S": f"{loss_D_S.item():.3f}",
            "D_R": f"{loss_D_R.item():.3f}",
            "cyc": f"{loss_cycle.item():.3f}",
            "id": f"{loss_id.item():.3f}",
        })

    # ---- epoch averages ----
    avg_loss_G = sum_loss_G / max(1, n_batches)
    avg_loss_D_S = sum_loss_D_S / max(1, n_batches)
    avg_loss_D_R = sum_loss_D_R / max(1, n_batches)
    avg_loss_GAN_S2R = sum_loss_GAN_S2R / max(1, n_batches)
    avg_loss_GAN_R2S = sum_loss_GAN_R2S / max(1, n_batches)
    avg_loss_cycle = sum_loss_cycle / max(1, n_batches)
    avg_loss_id = sum_loss_id / max(1, n_batches)

    # ---- append CSV ----
    with open(LOSS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), epoch,
            avg_loss_G, avg_loss_D_S, avg_loss_D_R,
            avg_loss_GAN_S2R, avg_loss_GAN_R2S,
            avg_loss_cycle, avg_loss_id
        ])

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

    # ======================
    # Save best checkpoint
    # ======================
    if avg_loss_G < best_loss_G:
        best_loss_G = avg_loss_G
        torch.save(G_S2R.state_dict(), BEST_DIR / "G_S2R_best.pth")
        torch.save(G_R2S.state_dict(), BEST_DIR / "G_R2S_best.pth")
        torch.save(D_S.state_dict(), BEST_DIR / "D_S_best.pth")
        torch.save(D_R.state_dict(), BEST_DIR / "D_R_best.pth")
        with open(BEST_DIR / "best.txt", "w") as f:
            f.write(f"best_epoch: {epoch}\n")
            f.write(f"best_loss_G: {best_loss_G}\n")

    epoch_time = time.time() - epoch_start
    with open(META_TXT, "a") as f:
        f.write(
            f"epoch {epoch} time_sec: {epoch_time:.1f}, "
            f"avg_loss_G: {avg_loss_G:.6f}, avg_loss_D_S: {avg_loss_D_S:.6f}, avg_loss_D_R: {avg_loss_D_R:.6f}\n"
        )

print("Training finished")
