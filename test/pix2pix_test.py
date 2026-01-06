import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import re
from pathlib import Path

# =========================
# Paths & Config
# =========================
DEFAULT_INPUT = "/home/nitzandu/synthetic-to-real-chessboard/data_test/synthetic_from_fen.png"
WEIGHTS_DIR = "/home/nitzandu/synthetic-to-real-chessboard/outputs/pix2pix_run_improved/weights"
DEFAULT_OUTDIR = "/home/nitzandu/synthetic-to-real-chessboard/data_test"
IMG_SIZE = 256

# =========================
# Blocks (Updated to match training)
# =========================
class UNetBlockDown(nn.Module):
    def __init__(self, in_c, out_c, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UNetBlockUp(nn.Module):
    def __init__(self, in_c, out_c, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# Generator (Updated to 7-layer depth)
# =========================
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.d1 = UNetBlockDown(3, 64, use_norm=False)
        self.d2 = UNetBlockDown(64, 128)
        self.d3 = UNetBlockDown(128, 256)
        self.d4 = UNetBlockDown(256, 512)
        self.d5 = UNetBlockDown(512, 512)
        self.d6 = UNetBlockDown(512, 512)
        self.d7 = UNetBlockDown(512, 512)

        # Decoder
        self.u1 = UNetBlockUp(512, 512, use_dropout=True)
        self.u2 = UNetBlockUp(1024, 512, use_dropout=True)
        self.u3 = UNetBlockUp(1024, 512, use_dropout=True)
        self.u4 = UNetBlockUp(1024, 256)
        self.u5 = UNetBlockUp(512, 128)
        self.u6 = UNetBlockUp(256, 64)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        en1 = self.d1(x)
        en2 = self.d2(en1)
        en3 = self.d3(en2)
        en4 = self.d4(en3)
        en5 = self.d5(en4)
        en6 = self.d6(en5)
        en7 = self.d7(en6)

        de1 = self.u1(en7)
        de2 = self.u2(torch.cat([de1, en6], 1))
        de3 = self.u3(torch.cat([de2, en5], 1))
        de4 = self.u4(torch.cat([de3, en4], 1))
        de5 = self.u5(torch.cat([de4, en3], 1))
        de6 = self.u6(torch.cat([de5, en2], 1))
        
        return self.out(torch.cat([de6, en1], 1))

# =========================
# Helper Functions
# =========================
def find_latest_generator(weights_dir):
    weights = list(Path(weights_dir).glob("G_*.pth"))
    if not weights:
        raise FileNotFoundError(f"No generator weights found in {weights_dir} (expected G_*.pth)")

    def step_num(p):
        m = re.search(r"G_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    latest = max(weights, key=step_num)
    return str(latest)

# =========================
# Main Inference
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input synthetic image")
    parser.add_argument("--weights", default=None, help="Path to G_*.pth (optional)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Where to save output")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Handle Weights
    if args.weights is None:
        weights_path = find_latest_generator(WEIGHTS_DIR)
        print(f"Using latest generator weights: {weights_path}")
    else:
        weights_path = args.weights

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    os.makedirs(args.outdir, exist_ok=True)

    # 2. Build & Load Model
    G = UNetGenerator().to(device)
    state = torch.load(weights_path, map_location=device)
    G.load_state_dict(state)
    G.eval()

    # 3. Preprocess
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    img = Image.open(args.input).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        fake = G(x)

    # 5. Save
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.outdir, f"{base}_pix2pix_improved.png")
    save_image(fake, out_path, normalize=True)

    print(f"DONE ✅ Saved to: {out_path}")

if __name__ == "__main__":
    main()