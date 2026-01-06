# Train_cyclegan.py
import itertools
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
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
SYN_DIR = DATA_ROOT / "synthetic"   # domain S (synthetic)
REAL_DIR = DATA_ROOT / "real"       # domain R (real)

OUT_DIR = Path("outputs/cyclegan_run1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 50
LR = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_ID = 0.0  # שימי 0.5 או 1.0 אם את רוצה Identity loss (לא חובה)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Dataset (Unpaired)
# ======================
class UnpairedImageDataset(Dataset):
    def __init__(self, synthetic_dir: Path, real_dir: Path, transform):
        # אפשר להרחיב פורמטים אם צריך
        self.synthetic = sorted(list(synthetic_dir.rglob("*.png")) + list(synthetic_dir.rglob("*.jpg")) + list(synthetic_dir.rglob("*.jpeg")))
        self.real = sorted(list(real_dir.rglob("*.jpg")) + list(real_dir.rglob("*.png")) + list(real_dir.rglob("*.jpeg")))
        self.transform = transform

        if len(self.synthetic) == 0:
            raise RuntimeError(f"No images found in synthetic_dir: {synthetic_dir}")
        if len(self.real) == 0:
            raise RuntimeError(f"No images found in real_dir: {real_dir}")

    def __len__(self):
        return max(len(self.synthetic), len(self.real))

    def __getitem__(self, idx):
        syn_path = self.synthetic[idx % len(self.synthetic)]
        real_path = self.real[idx % len(self.real)]

        syn_img = Image.open(syn_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")

        return self.transform(syn_img), self.transform(real_img)

# ======================
# Transforms
# ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # -> [-1, 1]
])

dataset = UnpairedImageDataset(SYN_DIR, REAL_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ======================
# Models
# ======================
cfg = NetConfig()

# Generators
G_S2R = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)  # Synthetic -> Real
G_R2S = build_generator("resnet", cfg, n_blocks=9).to(DEVICE)  # Real -> Synthetic

# Discriminators
D_S = build_discriminator(3, cfg).to(DEVICE)  # judges Synthetic domain
D_R = build_discriminator(3, cfg).to(DEVICE)  # judges Real domain

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
# Helpers: save samples
# ======================
def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)

@torch.no_grad()
def save_epoch_samples(G_s2r, G_r2s, fixed_syn_batch, out_dir: Path, epoch: int):
    """
    שומרת דוגמאות קבועות כדי להשוות איכות בין epochs.
    שומרת:
      - input synthetic
      - output fake real
      - reconstruction back to synthetic (אופציונלי אבל שימושי)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    G_s2r.eval()
    G_r2s.eval()

    syn = fixed_syn_batch.to(DEVICE)
    fake_real = G_s2r(syn)
    rec_syn = G_r2s(fake_real)

    # קבצים נפרדים וברורים
    save_image(denorm(syn),       out_dir / f"epoch{epoch:03d}_S_input.png", nrow=len(syn))
    save_image(denorm(fake_real), out_dir / f"epoch{epoch:03d}_R_fake.png",  nrow=len(syn))
    save_image(denorm(rec_syn),   out_dir / f"epoch{epoch:03d}_S_recon.png", nrow=len(syn))

    G_s2r.train()
    G_r2s.train()

# בוחרות סט דוגמאות קבוע פעם אחת (כדי להשוות תפוח לתפוח)
NUM_SAMPLES = 4
fixed_syn = torch.stack([dataset[i][0] for i in range(min(NUM_SAMPLES, len(dataset)))])
SAMPLES_DIR = OUT_DIR / "samples"

# ======================
# Training loop
# ======================
for epoch in range(1, EPOCHS + 1):
    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for syn_S, real_R in loop:
        syn_S = syn_S.to(DEVICE)   # domain S batch
        real_R = real_R.to(DEVICE) # domain R batch

        # ------------------
        # Train Generators
        # ------------------
        opt_G.zero_grad()

        # forward translations
        fake_R = G_S2R(syn_S)   # S -> fake R
        fake_S = G_R2S(real_R)  # R -> fake S

        # GAN loss (want D_* to think fakes are real)
        loss_GAN_S2R = gan_loss(D_R(fake_R), True)
        loss_GAN_R2S = gan_loss(D_S(fake_S), True)

        # cycle: S -> R -> S, and R -> S -> R
        rec_S = G_R2S(fake_R)
        rec_R = G_S2R(fake_S)
        loss_cycle = cycle_loss(rec_S, syn_S) + cycle_loss(rec_R, real_R)

        # identity loss (אופציונלי): אם מזינים כבר תמונה מהדומיין, הג'נרטור לא “יהרוס” אותה
        if LAMBDA_ID > 0:
            id_R = G_S2R(real_R)     # R fed into S->R generator
            id_S = G_R2S(syn_S)      # S fed into R->S generator
            loss_id = identity_loss(id_R, real_R) + identity_loss(id_S, syn_S)
        else:
            loss_id = torch.tensor(0.0, device=DEVICE)

        loss_G = loss_GAN_S2R + loss_GAN_R2S + (LAMBDA_CYCLE * loss_cycle) + (LAMBDA_ID * loss_id)
        loss_G.backward()
        opt_G.step()

        # ------------------
        # Train Discriminator D_R (Real domain)
        # ------------------
        opt_D_R.zero_grad()
        loss_D_R = (
            gan_loss(D_R(real_R), True) +
            gan_loss(D_R(fake_R.detach()), False)
        ) * 0.5
        loss_D_R.backward()
        opt_D_R.step()

        # ------------------
        # Train Discriminator D_S (Synthetic domain)
        # ------------------
        opt_D_S.zero_grad()
        loss_D_S = (
            gan_loss(D_S(syn_S), True) +
            gan_loss(D_S(fake_S.detach()), False)
        ) * 0.5
        loss_D_S.backward()
        opt_D_S.step()

        loop.set_postfix(
            G=float(loss_G.item()),
            D_R=float(loss_D_R.item()),
            D_S=float(loss_D_S.item()),
        )

    # ======================
    # Save epoch samples + checkpoints
    # ======================
    save_epoch_samples(G_S2R, G_R2S, fixed_syn, SAMPLES_DIR, epoch)

    torch.save(G_S2R.state_dict(), OUT_DIR / f"G_S2R_epoch{epoch}.pth")
    torch.save(G_R2S.state_dict(), OUT_DIR / f"G_R2S_epoch{epoch}.pth")
    torch.save(D_S.state_dict(),   OUT_DIR / f"D_S_epoch{epoch}.pth")
    torch.save(D_R.state_dict(),   OUT_DIR / f"D_R_epoch{epoch}.pth")

print("Training finished")
