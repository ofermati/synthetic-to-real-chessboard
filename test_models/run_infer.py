import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import re
import argparse

# =========================
# מודל
# =========================
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c, affine=False), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c, affine=False)
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

    def forward(self, x):
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.res(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.out(x)

def find_latest_weights(weights_dir: str) -> str:
    weights = list(Path(weights_dir).glob("epoch_*.pt"))
    if not weights:
        raise FileNotFoundError(f"No weights found in {weights_dir}")
    return str(max(weights, key=lambda p: int(re.search(r"epoch_(\d+)\.pt$", p.name).group(1))))

# =========================
# Helpers
# =========================
def tensor_to_pil(output_tensor_3chw: torch.Tensor) -> Image.Image:
    """Assumes tensor in [-1, 1], shape [3,H,W]."""
    img = (output_tensor_3chw * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(img.cpu())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_size", type=int, default=2048)
    parser.add_argument("--border_cut", type=int, default=0)

    # חדש:
    parser.add_argument("--mode", choices=["full", "tiled"], default="full",
                        help="full = run on whole image; tiled = split to grid and stitch back")
    parser.add_argument("--grid", type=int, default=8, help="grid size for tiled mode (e.g., 8 => 8x8 tiles)")
    args = parser.parse_args()

    INPUT_IMG_PATH = args.input
    WEIGHTS_DIR = args.weights_dir
    OUTPUT_PATH = args.output
    TARGET_SIZE = args.target_size
    BORDER_CUT = args.border_cut
    MODE = args.mode
    GRID = args.grid

    # CPU (כמו שהיה אצלך)
    device = "cpu"
    print(f"Using device: {device}")

    # 1) Load model
    weights_path = find_latest_weights(WEIGHTS_DIR)
    netG = ResnetGenerator().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["netG"] if isinstance(ckpt, dict) and "netG" in ckpt else ckpt
    netG.load_state_dict(state_dict)
    netG.eval()
    print(f"[INFO] Loaded weights: {weights_path}")

    # 2) Prepare image
    img = Image.open(INPUT_IMG_PATH).convert("RGB")
    w, h = img.size
    if BORDER_CUT > 0:
        img = img.crop((BORDER_CUT, BORDER_CUT, w - BORDER_CUT, h - BORDER_CUT))

    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    print(f"[INFO] Processing image size: {img.size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # =========================
    # MODE: full
    # =========================
    if MODE == "full":
        input_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
        print("[INFO] Running FULL inference...")
        with torch.no_grad():
            out = netG(input_tensor)  # [1,3,H,W]
        out_pil = tensor_to_pil(out.squeeze(0))
        out_pil.save(OUTPUT_PATH)
        print(f"✅ Saved FULL result to: {OUTPUT_PATH}")
        return

    # =========================
    # MODE: tiled (grid x grid)
    # =========================
    if TARGET_SIZE % GRID != 0:
        raise ValueError(f"target_size={TARGET_SIZE} must be divisible by grid={GRID} for clean tiling.")

    tile_size = TARGET_SIZE // GRID
    print(f"[INFO] Running TILED inference: grid={GRID}x{GRID}, tile_size={tile_size}")

    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE))

    with torch.no_grad():
        for r in range(GRID):
            for c in range(GRID):
                left = c * tile_size
                upper = r * tile_size
                right = left + tile_size
                lower = upper + tile_size

                tile = img.crop((left, upper, right, lower))  # PIL tile
                tile_tensor = transform(tile).unsqueeze(0).to(device)  # [1,3,t,t]

                out_tile = netG(tile_tensor).squeeze(0)  # [3,t,t]
                out_tile_pil = tensor_to_pil(out_tile)

                # stitch back in the original location
                canvas.paste(out_tile_pil, (left, upper))

                if (r * GRID + c) % max(1, (GRID * GRID // 8)) == 0:
                    print(f"[INFO] Processed tile ({r},{c})")

    canvas.save(OUTPUT_PATH)
    print(f"✅ Saved TILED stitched result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()