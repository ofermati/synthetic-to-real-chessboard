import os
import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


# -----------------------------
# 1) Datasets: recursive image loading
# -----------------------------
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under: {root_dir}")

        print(f"[INFO] Found {len(self.image_paths)} images under {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class UnpairedDataset(Dataset):
    """Return dict: A (synthetic), B (real). Randomly samples B."""
    def __init__(self, A: Dataset, B: Dataset):
        self.A = A
        self.B = B

    def __len__(self):
        return min(len(self.A), len(self.B))

    def __getitem__(self, idx: int):
        a = self.A[idx % len(self.A)]
        b = self.B[random.randint(0, len(self.B) - 1)]
        return {"A": a, "B": b}


# -----------------------------
# 2) Models: ResNet Generator + PatchGAN Discriminator
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0),
            nn.InstanceNorm2d(c, affine=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0),
            nn.InstanceNorm2d(c, affine=False)
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    CUT-style generator: encoder-decoder ResNet.
    Exposes intermediate feature maps for PatchNCE.
    """
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=9):
        super().__init__()

        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7, 1, 0),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 2, affine=False),
            nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 4, affine=False),
            nn.ReLU(True),
        )

        self.res = nn.Sequential(*[ResBlock(ngf * 4) for _ in range(n_blocks)])

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2, affine=False),
            nn.ReLU(True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, 7, 1, 0),
            nn.Tanh()
        )

        self.nce_layers = ["enc0", "enc1", "enc2", "res"]

    def encode_features(self, x) -> Dict[str, torch.Tensor]:
        feats = {}
        x0 = self.enc0(x); feats["enc0"] = x0
        x1 = self.enc1(x0); feats["enc1"] = x1
        x2 = self.enc2(x1); feats["enc2"] = x2
        xr = self.res(x2); feats["res"] = xr
        return feats

    def forward(self, x, return_feats: bool = False):
        feats = self.encode_features(x)
        xr = feats["res"]
        x = self.dec2(xr)
        x = self.dec1(x)
        y = self.out(x)
        if return_feats:
            return y, feats
        return y


class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=3, ndf=64, n_layers=3):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_c, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False),
            nn.LeakyReLU(0.2, True),
        ]

        layers += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -----------------------------
# 3) CUT: Patch sampling + NCE loss
# -----------------------------
class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, mlp_dim=256):
        super().__init__()
        self.use_mlp = use_mlp
        self.mlp_dim = mlp_dim
        self.mlps = nn.ModuleDict()  # created lazily per layer channel size

    def _make_mlp(self, name: str, in_dim: int, device):
        mlp = nn.Sequential(
            nn.Linear(in_dim, self.mlp_dim),
            nn.ReLU(True),
            nn.Linear(self.mlp_dim, self.mlp_dim)
        ).to(device)
        self.mlps[name] = mlp

    def forward(self, feats: Dict[str, torch.Tensor], n_patches: int, patch_ids=None):
        out = {}
        ids_out = {}
        for name, f in feats.items():
            B, C, H, W = f.shape
            f_ = f.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,HW,C]

            if patch_ids is None or name not in patch_ids:
                n = min(n_patches, H * W)
                idx = torch.randperm(H * W, device=f.device)[:n]
                idx = idx.unsqueeze(0).repeat(B, 1)  # [B,n]
            else:
                idx = patch_ids[name]  # [B,n]

            idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)
            patches = torch.gather(f_, dim=1, index=idx_exp)  # [B,n,C]
            patches = patches.reshape(B * patches.shape[1], C)  # [B*n, C]

            if self.use_mlp:
                if name not in self.mlps:
                    self._make_mlp(name, C, device=f.device)
                patches = self.mlps[name](patches)

            patches = F.normalize(patches, dim=1)
            out[name] = patches
            ids_out[name] = idx

        return out, ids_out


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        logits = torch.mm(q, k.t()) / self.t
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


class GANLossLS(nn.Module):
    def forward(self, pred, target_is_real: bool):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


# -----------------------------
# 4) Train config + paths
# -----------------------------
@dataclass
class Config:
    synthetic_root: str = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/synthetic"
    real_root: str      = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/real"

    outputs_root: str   = "/home/nitzandu/synthetic-to-real-chessboard/outputs"
    run_name: str       = "cut_1_8X8"

    img_size: int       = 256
    batch_size: int     = 1
    num_workers: int    = 4

    epochs: int         = 100
    lr: float           = 1e-4
    beta1: float        = 0.5
    beta2: float        = 0.999

    nce_layers: Tuple[str, ...] = ("enc0", "enc1", "enc2", "res")
    nce_weight: float   = 4.0
    nce_num_patches: int = 256
    nce_temperature: float = 0.07

    # identity/self-regularization
    use_identity: bool  = False
    id_weight: float    = 0.0

    # augmentation
    use_color_jitter: bool = True
    # NOTE: flip is off by default for stability
    use_hflip: bool = False

    device: str         = "cuda" if torch.cuda.is_available() else "cpu"

    resume: bool        = True
    resume_ckpt_path: str = ""


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_dirs(cfg: Config) -> Tuple[Path, Path]:
    run_dir = Path(cfg.outputs_root) / cfg.run_name
    images_dir = run_dir / "images"
    weights_dir = run_dir / "weights"
    ensure_dir(images_dir)
    ensure_dir(weights_dir)
    return images_dir, weights_dir


def latest_checkpoint(weights_dir: Path) -> Optional[Path]:
    pts = list(weights_dir.glob("epoch_*.pt"))
    if not pts:
        return None

    def epoch_num(p: Path) -> int:
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    pts.sort(key=epoch_num)
    return pts[-1] if epoch_num(pts[-1]) >= 0 else None


def save_samples(epoch: int, real_A, fake_B, real_B, images_dir: Path):
    def to01(x): return (x * 0.5 + 0.5).clamp(0, 1)
    grid = torch.cat([to01(real_A), to01(fake_B), to01(real_B)], dim=0)  # A | G(A) | B
    save_image(grid, str(images_dir / f"epoch_{epoch:03d}.png"), nrow=3)


def save_ckpt(epoch: int, netG, netD, netF, optG, optD, optF, weights_dir: Path, cfg: Config):
    ckpt = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "netF": netF.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "optF": None if optF is None else optF.state_dict(),
        "cfg": cfg.__dict__,
    }
    torch.save(ckpt, str(weights_dir / f"epoch_{epoch:03d}.pt"))
    torch.save(ckpt, str(weights_dir / "latest.pt"))


def load_ckpt(path: Path, netG, netD, netF, optG, optD, optF, device: str) -> int:
    ckpt = torch.load(str(path), map_location=device)
    netG.load_state_dict(ckpt["netG"])
    netD.load_state_dict(ckpt["netD"])
    netF.load_state_dict(ckpt["netF"])
    optG.load_state_dict(ckpt["optG"])
    optD.load_state_dict(ckpt["optD"])
    if optF is not None and ckpt.get("optF") is not None:
        optF.load_state_dict(ckpt["optF"])
    start_epoch = int(ckpt["epoch"]) + 1
    print(f"[INFO] Resumed from {path} (next epoch: {start_epoch})")
    return start_epoch


# -----------------------------
# 5) Train loop
# -----------------------------
def main():
    cfg = Config()
    images_dir, weights_dir = run_dirs(cfg)
    print(f"[INFO] Outputs:")
    print(f"  images : {images_dir}")
    print(f"  weights: {weights_dir}")
    print(f"[INFO] device: {cfg.device}")

    # transforms
    tfm_list = [transforms.Resize((cfg.img_size, cfg.img_size))]

    if cfg.use_color_jitter:
        tfm_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02))

    if cfg.use_hflip:
        tfm_list.append(transforms.RandomHorizontalFlip(p=0.5))

    tfm_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    tfm = transforms.Compose(tfm_list)

    A_ds = RecursiveImageDataset(cfg.synthetic_root, tfm)
    B_ds = RecursiveImageDataset(cfg.real_root, tfm)
    train_ds = UnpairedDataset(A_ds, B_ds)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    netG = ResnetGenerator().to(cfg.device)
    netD = PatchDiscriminator().to(cfg.device)
    netF = PatchSampleF(use_mlp=True, mlp_dim=256).to(cfg.device)

    gan_loss = GANLossLS().to(cfg.device)
    nce_loss_fn = PatchNCELoss(temperature=cfg.nce_temperature).to(cfg.device)

    optG = torch.optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optF = None  # created after first forward (because MLPs are lazy)

    def compute_nce(real_A, fake_B):
        # ✅ IMPORTANT: encoder features only (no G(G(A))!)
        feats_A = netG.encode_features(real_A)
        feats_B = netG.encode_features(fake_B)

        feats_A = {k: feats_A[k] for k in cfg.nce_layers}
        feats_B = {k: feats_B[k] for k in cfg.nce_layers}

        q, ids = netF(feats_B, n_patches=cfg.nce_num_patches, patch_ids=None)
        k, _   = netF(feats_A, n_patches=cfg.nce_num_patches, patch_ids=ids)

        total = 0.0
        for layer in cfg.nce_layers:
            total = total + nce_loss_fn(q[layer], k[layer].detach())  # ✅ detach stabilizes
        return total / len(cfg.nce_layers)

    # -----------------------------------
    # Resume logic
    # -----------------------------------
    start_epoch = 1
    if cfg.resume:
        ckpt_path = None
        if cfg.resume_ckpt_path.strip():
            ckpt_path = Path(cfg.resume_ckpt_path.strip())
        else:
            if (weights_dir / "latest.pt").exists():
                ckpt_path = weights_dir / "latest.pt"
            else:
                ckpt_path = latest_checkpoint(weights_dir)

        if ckpt_path is not None and ckpt_path.exists():
            # instantiate netF mlps via dummy nce pass
            batch0 = next(iter(loader))
            real_A0 = batch0["A"].to(cfg.device)
            with torch.no_grad():
                fake_B0 = netG(real_A0)
            _ = compute_nce(real_A0, fake_B0)  # creates mlps
            optF = torch.optim.Adam(netF.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

            start_epoch = load_ckpt(ckpt_path, netG, netD, netF, optG, optD, optF, cfg.device)
        else:
            print("[INFO] No checkpoint found to resume from. Starting fresh.")

    step = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        netG.train(); netD.train(); netF.train()
        running_g = 0.0
        running_d = 0.0
        running_nce = 0.0
        running_id = 0.0

        for batch in loader:
            real_A = batch["A"].to(cfg.device, non_blocking=True)  # synthetic
            real_B = batch["B"].to(cfg.device, non_blocking=True)  # real

            # -----------------
            # Update D
            # -----------------
            with torch.no_grad():
                fake_B = netG(real_A)

            pred_real = netD(real_B)
            pred_fake = netD(fake_B.detach())
            loss_D = 0.5 * (gan_loss(pred_real, True) + gan_loss(pred_fake, False))

            optD.zero_grad(set_to_none=True)
            loss_D.backward()
            optD.step()

            # -----------------
            # Update G (+F)
            # -----------------
            fake_B = netG(real_A)
            pred_fake_for_G = netD(fake_B)
            loss_G_gan = gan_loss(pred_fake_for_G, True)

            loss_G_nce = compute_nce(real_A, fake_B) * cfg.nce_weight

            loss_id = torch.tensor(0.0, device=cfg.device)
            if cfg.use_identity:
                id_B = netG(real_B)
                loss_id = F.l1_loss(id_B, real_B) * cfg.id_weight

            loss_G = loss_G_gan + loss_G_nce + loss_id

            if optF is None:
                optF = torch.optim.Adam(netF.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

            optG.zero_grad(set_to_none=True)
            optF.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()
            optF.step()

            running_g += loss_G.item()
            running_d += loss_D.item()
            running_nce += loss_G_nce.item()
            running_id += loss_id.item()
            step += 1

            if step % 100 == 0:
                print(f"[E{epoch:03d} step {step}] "
                      f"D: {running_d/100:.4f} | G: {running_g/100:.4f} | "
                      f"NCE: {running_nce/100:.4f} | ID: {running_id/100:.4f}")
                running_g = running_d = running_nce = running_id = 0.0

        # -----------------
        # Save sample + ckpt each epoch
        # -----------------
        netG.eval()
        with torch.no_grad():
            fake_B_vis = netG(real_A[:1])
        save_samples(epoch, real_A[:1], fake_B_vis[:1], real_B[:1], images_dir)

        save_ckpt(epoch, netG, netD, netF, optG, optD, optF, weights_dir, cfg)
        print(f"[INFO] Saved: images/epoch_{epoch:03d}.png and weights/epoch_{epoch:03d}.pt")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    main()
