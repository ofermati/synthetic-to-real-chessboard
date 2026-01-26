import sys
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.networks import NetConfig, build_generator


# =========================
# CONFIG - תשני רק כאן
# =========================
SYN_ROOT   = Path("temp_data/zoomed")      # synthetic zoomed folders (game2/frame_200/*.png)
REAL_ROOT  = Path("datasets")      # real frames (Game2/frame_000200.jpg)
GAME       = "game2"                      # "game2", "game7", etc.

WEIGHTS_PATH = Path("outputs/cyclegan_run1/G_S2R_epoch50.pth")

OUT_ROOT   = Path("outputs/eval_zoomed")  # where BEST_* outputs go
SAVE_BEST_ONLY = True                     # True: save only BEST per frame, False: save all views too

# tiling (same as your single-image script)
BORDER_CUT  = 0
TARGET_SIZE = 2048
TILE_SIZE   = 256

# evaluation / selection
SELECT_METRIC = "ssim"   # "ssim" | "psnr" | "l1" | "mse"
MAX_FRAMES = 150         # None = run all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================


def try_import_ssim():
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim
    except Exception:
        return None


def load_generator(weights_path: Path, device: str):
    cfg = NetConfig()
    netG = build_generator("resnet", cfg, n_blocks=9).to(device)
    state = torch.load(weights_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()
    return netG


def to_tensor_norm():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])


def run_tiled_inference(netG, img_pil: Image.Image, target_size: int, tile_size: int, device: str):
    if target_size % tile_size != 0:
        raise ValueError("TARGET_SIZE must be divisible by TILE_SIZE")

    img = img_pil.resize((target_size, target_size), Image.BICUBIC)
    tfm = to_tensor_norm()
    out_full = Image.new("RGB", (target_size, target_size))

    with torch.no_grad():
        for y in range(0, target_size, tile_size):
            for x in range(0, target_size, tile_size):
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                inp = tfm(tile).unsqueeze(0).to(device)

                out = netG(inp).squeeze(0).cpu()         # [-1,1]
                out = (out * 0.5 + 0.5).clamp(0, 1)      # [0,1]

                pil_tile = transforms.ToPILImage()(out)
                out_full.paste(pil_tile, (x, y))

    return out_full


def pil_to_tensor_01(img: Image.Image):
    return transforms.ToTensor()(img).unsqueeze(0).clamp(0, 1)


def compute_metrics(pred_pil: Image.Image, real_pil: Image.Image, ssim_fn):
    if real_pil.size != pred_pil.size:
        real_pil = real_pil.resize(pred_pil.size, Image.BICUBIC)

    pred = pil_to_tensor_01(pred_pil)
    real = pil_to_tensor_01(real_pil)

    l1 = F.l1_loss(pred, real).item()
    mse = F.mse_loss(pred, real).item()
    psnr = float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    ssim_val = None
    if ssim_fn is not None:
        import numpy as np
        p = (np.array(pred_pil).astype("float32") / 255.0)
        g = (np.array(real_pil).astype("float32") / 255.0)
        ssim_val = float(ssim_fn(p, g, channel_axis=2, data_range=1.0))

    return {"l1": l1, "mse": mse, "psnr": psnr, "ssim": ssim_val}


def frame_num_from_real_name(name: str):
    m = re.search(r"frame_(\d+)\.", name)
    return int(m.group(1)) if m else None


def main():
    if not SYN_ROOT.exists():
        raise SystemExit(f"SYN_ROOT not found: {SYN_ROOT}")
    if not REAL_ROOT.exists():
        raise SystemExit(f"REAL_ROOT not found: {REAL_ROOT}")
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")

    ssim_fn = try_import_ssim()
    metric = SELECT_METRIC
    if metric == "ssim" and ssim_fn is None:
        print("Note: scikit-image not installed, SSIM selection disabled. Install: pip install scikit-image")
        print("Falling back to PSNR.")
        metric = "psnr"

    print("Device:", DEVICE)
    netG = load_generator(WEIGHTS_PATH, DEVICE)
    print("Loaded weights:", WEIGHTS_PATH)

    # build paths
    game_syn = SYN_ROOT / GAME.lower()                # temp_data/zoomed/game2
    game_real = REAL_ROOT / ("Game" + re.sub(r"\D", "", GAME))  # datasets/cut_8X8/Game2

    if not game_syn.exists():
        raise SystemExit(f"game syn not found: {game_syn}")
    if not game_real.exists():
        raise SystemExit(f"game real not found: {game_real}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    real_frames = sorted(game_real.glob("frame_*.jpg"))
    if MAX_FRAMES is not None:
        real_frames = real_frames[:MAX_FRAMES]

    per_kind = {}
    overall = {"count": 0, "l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0, "ssim_count": 0}
    chosen_kind_count = {}

    def add_metric(bucket, m):
        bucket["count"] += 1
        bucket["l1"] += m["l1"]
        bucket["mse"] += m["mse"]
        bucket["psnr"] += m["psnr"]
        if m["ssim"] is not None:
            bucket["ssim"] += m["ssim"]
            bucket["ssim_count"] += 1

    t0 = time.time()

    for real_path in tqdm(real_frames, desc="Evaluating frames"):
        frame_num = frame_num_from_real_name(real_path.name)
        if frame_num is None:
            continue

        syn_dir = game_syn / f"frame_{frame_num}"
        if not syn_dir.exists():
            continue

        syn_imgs = sorted(list(syn_dir.glob("*.png")))
        if not syn_imgs:
            continue

        real_img = Image.open(real_path).convert("RGB")

        candidates = []
        for syn_path in syn_imgs:
            syn_img = Image.open(syn_path).convert("RGB")

            if BORDER_CUT > 0:
                w, h = syn_img.size
                bc = BORDER_CUT
                syn_img = syn_img.crop((bc, bc, w - bc, h - bc))

            pred = run_tiled_inference(netG, syn_img, TARGET_SIZE, TILE_SIZE, DEVICE)
            m = compute_metrics(pred, real_img, ssim_fn)

            kind = syn_path.stem.split("_", 1)[-1] if "_" in syn_path.stem else syn_path.stem
            candidates.append((syn_path, kind, pred, m))

            if kind not in per_kind:
                per_kind[kind] = {"count": 0, "l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0, "ssim_count": 0}
            add_metric(per_kind[kind], m)

            if not SAVE_BEST_ONLY:
                out_path = OUT_ROOT / GAME.lower() / f"frame_{frame_num:06d}" / f"{kind}_pred.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pred.save(out_path)

        if not candidates:
            continue

        if metric in ("ssim", "psnr"):
            best = max(candidates, key=lambda x: (-1e9 if x[3][metric] is None else x[3][metric]))
        else:
            best = min(candidates, key=lambda x: x[3][metric])

        _, best_kind, best_pred, best_m = best
        add_metric(overall, best_m)
        chosen_kind_count[best_kind] = chosen_kind_count.get(best_kind, 0) + 1

        out_best = OUT_ROOT / GAME.lower() / f"frame_{frame_num:06d}" / f"BEST_{best_kind}.png"
        out_best.parent.mkdir(parents=True, exist_ok=True)
        best_pred.save(out_best)

    dt = time.time() - t0

    print("\n=== Done ===")
    print(f"Total time: {dt:.2f}s")
    if overall["count"] == 0:
        print("No matched frames were evaluated.")
        return

    print(f"\nSelection metric: {metric}")
    print(f"Evaluated frames (best-of-views): {overall['count']}")

    print("\nBest-of-views Averages:")
    print(f"  L1   : {overall['l1'] / overall['count']:.6f}")
    print(f"  MSE  : {overall['mse'] / overall['count']:.6f}")
    print(f"  PSNR : {overall['psnr'] / overall['count']:.3f} dB")
    if overall["ssim_count"] > 0:
        print(f"  SSIM : {overall['ssim'] / overall['ssim_count']:.4f}")
    else:
        print("  SSIM : skipped")

    print("\nWhich view was chosen most often:")
    for k, v in sorted(chosen_kind_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nSaved outputs to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
