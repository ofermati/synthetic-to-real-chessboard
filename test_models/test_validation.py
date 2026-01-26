import argparse
import random
import re
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# -----------------------------
# Datasets
# -----------------------------
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
# Models (same as training)
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
            nn.InstanceNorm2d(c, affine=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
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
            nn.Tanh(),
        )

    def encode_features(self, x) -> Dict[str, torch.Tensor]:
        feats = {}
        x0 = self.enc0(x); feats["enc0"] = x0
        x1 = self.enc1(x0); feats["enc1"] = x1
        x2 = self.enc2(x1); feats["enc2"] = x2
        xr = self.res(x2); feats["res"] = xr
        return feats

    def forward(self, x):
        feats = self.encode_features(x)
        x = self.dec2(feats["res"])
        x = self.dec1(x)
        return self.out(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_c, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
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


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, mlp_dim=256):
        super().__init__()
        self.use_mlp = use_mlp
        self.mlp_dim = mlp_dim
        self.mlps = nn.ModuleDict()

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
            f_ = f.permute(0, 2, 3, 1).reshape(B, H * W, C)

            if patch_ids is None or name not in patch_ids:
                n = min(n_patches, H * W)
                idx = torch.randperm(H * W, device=f.device)[:n]
                idx = idx.unsqueeze(0).repeat(B, 1)
            else:
                idx = patch_ids[name]

            idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)
            patches = torch.gather(f_, dim=1, index=idx_exp)  # [B,n,C]
            patches = patches.reshape(B * patches.shape[1], C)

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


def find_ckpt(ckpt_path: str) -> Path:
    p = Path(ckpt_path)
    if p.is_file():
        return p
    if p.is_dir():
        if (p / "latest.pt").exists():
            return p / "latest.pt"
        pts = list(p.glob("epoch_*.pt"))
        if pts:
            def epoch_num(x: Path) -> int:
                m = re.search(r"epoch_(\d+)\.pt$", x.name)
                return int(m.group(1)) if m else -1
            return max(pts, key=epoch_num)
    raise FileNotFoundError(f"Could not find checkpoint at: {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_synth_root", required=True, help="Game 7 synthetic root")
    ap.add_argument("--real_root", required=True, help="Real images root (for unpaired B sampling)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint file OR weights dir")
    ap.add_argument("--out_csv", default="val_losses_game7.csv")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--nce_weight", type=float, default=2.0)
    ap.add_argument("--nce_num_patches", type=int, default=256)
    ap.add_argument("--nce_temperature", type=float, default=0.07)
    ap.add_argument("--use_identity", action="store_true")
    ap.add_argument("--id_weight", type=float, default=0.5)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # transforms (NO augmentation in validation)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    A_val = RecursiveImageDataset(args.val_synth_root, tfm)
    B_real = RecursiveImageDataset(args.real_root, tfm)
    val_ds = UnpairedDataset(A_val, B_real)

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    netG = ResnetGenerator().to(device)
    netD = PatchDiscriminator().to(device)
    netF = PatchSampleF(use_mlp=True, mlp_dim=256).to(device)

    gan_loss = GANLossLS().to(device)
    nce_loss_fn = PatchNCELoss(temperature=args.nce_temperature).to(device)

    ckpt_path = find_ckpt(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    netG.load_state_dict(ckpt["netG"])
    netD.load_state_dict(ckpt["netD"])
    netF.load_state_dict(ckpt["netF"])

    netG.eval(); netD.eval(); netF.eval()

    def compute_nce(real_A, fake_B):
        feats_A = netG.encode_features(real_A)
        feats_B = netG.encode_features(fake_B)
        layers = ("enc0", "enc1", "enc2", "res")

        q, ids = netF({k: feats_B[k] for k in layers}, n_patches=args.nce_num_patches, patch_ids=None)
        k, _   = netF({k: feats_A[k] for k in layers}, n_patches=args.nce_num_patches, patch_ids=ids)

        total = 0.0
        for layer in layers:
            total = total + nce_loss_fn(q[layer], k[layer].detach())
        return total / len(layers)

    # accumulate
    sum_g = sum_d = sum_nce = sum_id = 0.0
    n = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            real_A = batch["A"].to(device, non_blocking=True)  # synthetic
            real_B = batch["B"].to(device, non_blocking=True)  # random real

            fake_B = netG(real_A)

            # D loss (same formula as training)
            pred_real = netD(real_B)
            pred_fake = netD(fake_B)
            loss_D = 0.5 * (gan_loss(pred_real, True) + gan_loss(pred_fake, False))

            # G losses
            pred_fake_for_G = netD(fake_B)
            loss_G_gan = gan_loss(pred_fake_for_G, True)

            loss_G_nce = compute_nce(real_A, fake_B) * args.nce_weight

            loss_id = torch.tensor(0.0, device=device)
            if args.use_identity:
                id_B = netG(real_B)
                loss_id = F.l1_loss(id_B, real_B) * args.id_weight

            loss_G = loss_G_gan + loss_G_nce + loss_id

            sum_g += float(loss_G.item())
            sum_d += float(loss_D.item())
            sum_nce += float(loss_G_nce.item())
            sum_id += float(loss_id.item())
            n += 1

            if batch_idx % 50 == 0:
                print(f"[VAL] {batch_idx}: G={sum_g/n:.4f} D={sum_d/n:.4f} NCE={sum_nce/n:.4f} ID={sum_id/n:.4f}")

            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

    avg_g = sum_g / max(n, 1)
    avg_d = sum_d / max(n, 1)
    avg_nce = sum_nce / max(n, 1)
    avg_id = sum_id / max(n, 1)

    print("[DONE] Validation losses on Game7:")
    print(f"  avg_loss_G  : {avg_g:.6f}")
    print(f"  avg_loss_D  : {avg_d:.6f}")
    print(f"  avg_loss_NCE: {avg_nce:.6f}")
    print(f"  avg_loss_ID : {avg_id:.6f}")

    print(f"[INFO] wrote: {out_csv}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()