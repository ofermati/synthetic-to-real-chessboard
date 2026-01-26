import random
import re
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# =========================
# FIXED CONFIG (no CLI args)
# =========================
VAL_SYNTH_ROOT = "temp_data/zoomed/game7"                  # Game 7 synthetic
REAL_ROOT      = "datasets/unpaired/real/Game7"            # unused here (kept for reference)
CKPT_PATH      = "outputs/cut_1_8X8_new/weights/latest.pt" # OR a dir with latest.pt / epoch_*.pt

IMG_SIZE        = 256
BATCH_SIZE      = 1
NUM_WORKERS     = 0
NCE_WEIGHT      = 2.0
NCE_NUM_PATCHES = 256
NCE_TEMPERATURE = 0.07
MAX_BATCHES     = 0   # 0 = no limit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Dataset: just synthetic images
# -----------------------------
class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ]
        if not self.image_paths:
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


class PatchSampleF(nn.Module):
    """
    Note: MLPs are created lazily (during forward) based on feature channel dims.
    So before loading a checkpoint that contains mlps.*, we must create them first.
    """
    def __init__(self, use_mlp=True, mlp_dim=256):
        super().__init__()
        self.use_mlp = use_mlp
        self.mlp_dim = mlp_dim
        self.mlps = nn.ModuleDict()

    def _make_mlp(self, name: str, in_dim: int, device):
        self.mlps[name] = nn.Sequential(
            nn.Linear(in_dim, self.mlp_dim),
            nn.ReLU(True),
            nn.Linear(self.mlp_dim, self.mlp_dim),
        ).to(device)

    def forward(self, feats: Dict[str, torch.Tensor], n_patches: int, patch_ids=None):
        out, ids_out = {}, {}
        for name, f in feats.items():
            B, C, H, W = f.shape
            f_ = f.permute(0, 2, 3, 1).reshape(B, H * W, C)

            if patch_ids is None or name not in patch_ids:
                n = min(n_patches, H * W)
                idx = torch.randperm(H * W, device=f.device)[:n]
                idx = idx.unsqueeze(0).repeat(B, 1)
            else:
                idx = patch_ids[name]

            patches = torch.gather(f_, 1, idx.unsqueeze(-1).expand(-1, -1, C))
            patches = patches.reshape(B * patches.shape[1], C)

            if self.use_mlp:
                if name not in self.mlps:
                    self._make_mlp(name, C, device=f.device)
                patches = self.mlps[name](patches)

            out[name] = F.normalize(patches, dim=1)
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


def build_netF_mlps_before_loading(netG: ResnetGenerator, netF: PatchSampleF, device: str, img_size: int):
    """
    Create the lazy MLP modules inside netF (mlps.enc0 / enc1 / enc2 / res)
    BEFORE loading checkpoint weights, so load_state_dict won't complain.
    """
    layers = ("enc0", "enc1", "enc2", "res")
    netG.eval()
    netF.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        feats = netG.encode_features(dummy)
        _ = netF({k: feats[k] for k in layers}, n_patches=1, patch_ids=None)


def main():
    print(f"[INFO] device: {DEVICE}")

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    ds = RecursiveImageDataset(VAL_SYNTH_ROOT, tfm)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )

    netG = ResnetGenerator().to(DEVICE)
    netF = PatchSampleF(use_mlp=True, mlp_dim=256).to(DEVICE)
    nce_loss_fn = PatchNCELoss(temperature=NCE_TEMPERATURE).to(DEVICE)

    ckpt_file = find_ckpt(CKPT_PATH)
    print(f"[INFO] Loading checkpoint: {ckpt_file}")

    ckpt = torch.load(str(ckpt_file), map_location=DEVICE)

    # Load netG first (static architecture)
    if "netG" not in ckpt:
        raise KeyError("Checkpoint missing key 'netG'")
    netG.load_state_dict(ckpt["netG"])

    # IMPORTANT: build netF's MLP modules BEFORE loading netF weights
    build_netF_mlps_before_loading(netG, netF, DEVICE, IMG_SIZE)

    if "netF" not in ckpt:
        raise KeyError("Checkpoint missing key 'netF'")
    netF.load_state_dict(ckpt["netF"])

    netG.eval()
    netF.eval()

    layers = ("enc0", "enc1", "enc2", "res")

    def compute_nce(x, y):
        feats_x = netG.encode_features(x)
        feats_y = netG.encode_features(y)

        q, ids = netF({k: feats_y[k] for k in layers}, n_patches=NCE_NUM_PATCHES, patch_ids=None)
        k, _   = netF({k: feats_x[k] for k in layers}, n_patches=NCE_NUM_PATCHES, patch_ids=ids)

        total = 0.0
        for layer in layers:
            total = total + nce_loss_fn(q[layer], k[layer].detach())
        return total / len(layers)

    total_nce = 0.0
    n = 0

    with torch.no_grad():
        for i, real_A in enumerate(loader, 1):
            real_A = real_A.to(DEVICE, non_blocking=True)
            fake_B = netG(real_A)

            nce = compute_nce(real_A, fake_B) * NCE_WEIGHT
            total_nce += float(nce.item())
            n += 1

            if i % 50 == 0:
                print(f"[VAL] {i}: avg_NCE={total_nce/n:.4f}")

            if MAX_BATCHES > 0 and i >= MAX_BATCHES:
                break

    print("[DONE] Game7 NCE-only:")
    print(f"  count    : {n}")
    print(f"  avg_NCE  : {total_nce/max(n,1):.6f}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()