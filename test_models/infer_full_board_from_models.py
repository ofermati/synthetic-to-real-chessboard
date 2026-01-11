from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.networks import NetConfig, build_generator

# =========================
# CONFIG - תשני רק כאן
# =========================
INPUT_IMAGE = Path("temp_data/zoomed/game7/frame_4952/2_west.png")
WEIGHTS_PATH = Path("outputs/cyclegan_run1/G_S2R_epoch21.pth")
OUTPUT_IMAGE = Path("outputs/infer/board_full.png")

BORDER_CUT = 0
TARGET_SIZE = 2048
TILE_SIZE = 256
# =========================

def load_generator(weights_path, device):
    cfg = NetConfig()
    netG = build_generator("resnet", cfg, n_blocks=9).to(device)

    state = torch.load(weights_path, map_location=device)
    netG.load_state_dict(state)

    netG.eval()
    return netG


def main():
    if not INPUT_IMAGE.exists():
        raise SystemExit(f"Input not found: {INPUT_IMAGE}")
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"Weights not found: {WEIGHTS_PATH}")
    if TARGET_SIZE % TILE_SIZE != 0:
        raise SystemExit("TARGET_SIZE must be divisible by TILE_SIZE")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    netG = load_generator(WEIGHTS_PATH, device)
    print("Loaded weights:", WEIGHTS_PATH)

    img = Image.open(INPUT_IMAGE).convert("RGB")
    w, h = img.size

    if BORDER_CUT > 0:
        img = img.crop((BORDER_CUT, BORDER_CUT, w - BORDER_CUT, h - BORDER_CUT))

    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    final_image = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE))

    with torch.no_grad():
        for y in range(0, TARGET_SIZE, TILE_SIZE):
            for x in range(0, TARGET_SIZE, TILE_SIZE):
                tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
                inp = transform(tile).unsqueeze(0).to(device)

                out = netG(inp).squeeze(0).cpu()
                out = (out * 0.5 + 0.5).clamp(0, 1)

                pil_tile = transforms.ToPILImage()(out)
                final_image.paste(pil_tile, (x, y))

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(OUTPUT_IMAGE)
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
