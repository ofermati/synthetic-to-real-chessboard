import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# כדי ש- "from models..." יעבוד גם כשמריצים את הקובץ ישירות
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.networks import NetConfig, build_generator  # noqa: E402


# =========================
# Paths (יחסיים לשורש הפרויקט)
# =========================
INPUT_IMG_PATH = PROJECT_ROOT / "temp_data/zoomed/game5/frame_1444/2_west.png"
WEIGHTS_PATH   = PROJECT_ROOT / "outputs/cyclegan_run1/G_S2R_epoch50.pth"
OUTPUT_PATH    = PROJECT_ROOT / "outputs/full_cpu_result_.png"

TARGET_SIZE = 2048
BORDER_CUT = 0
# =========================


def load_generator(weights_path: Path, device: str):
    cfg = NetConfig()
    netG = build_generator("resnet", cfg, n_blocks=9).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    # תומך גם ב-ckpt עם netG וגם ב-state_dict ישיר
    state_dict = ckpt["netG"] if isinstance(ckpt, dict) and "netG" in ckpt else ckpt

    netG.load_state_dict(state_dict, strict=True)
    netG.eval()
    return netG


def main():
    device = "cpu"
    print(f"Using device: {device}")

    if not INPUT_IMG_PATH.exists():
        raise SystemExit(f"Input not found: {INPUT_IMG_PATH}")
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"Weights not found: {WEIGHTS_PATH}")

    netG = load_generator(WEIGHTS_PATH, device)
    print(f"Loaded weights: {WEIGHTS_PATH}")

    img = Image.open(INPUT_IMG_PATH).convert("RGB")
    w, h = img.size

    if BORDER_CUT > 0:
        img = img.crop((BORDER_CUT, BORDER_CUT, w - BORDER_CUT, h - BORDER_CUT))

    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    print(f"[INFO] Processing full image size: {img.size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    print("[INFO] Running inference on full image... please wait...")
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    output_img = output_tensor.squeeze(0).cpu()
    output_img = (output_img * 0.5 + 0.5).clamp(0, 1)
    final_pil = transforms.ToPILImage()(output_img)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_pil.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
