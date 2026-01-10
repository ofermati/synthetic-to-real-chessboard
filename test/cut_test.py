import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import re

# =========================
# Paths
# =========================
INPUT_IMG_PATH = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/synthetic/Game4/frame_640/G4_640_e_r6_c0.png"
WEIGHTS_DIR = "/home/nitzandu/synthetic-to-real-chessboard/outputs/cut_1_8X8/weights"
OUTPUT_PATH = "/home/nitzandu/synthetic-to-real-chessboard/data_test/output_cut_1_8X8_6.png"
IMG_SIZE = 256

# =========================
# Updated Model Architecture (Exact match to training)
# =========================
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
    def forward(self, x): return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=9):
        super().__init__()
        # Encoder
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

        # Residual Blocks
        self.res = nn.Sequential(*[ResBlock(ngf * 4) for _ in range(n_blocks)])

        # Decoder
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

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        xr = self.res(x2)
        d2 = self.dec2(xr)
        d1 = self.dec1(d2)
        return self.out(d1)

# =========================
# Helper Functions
# =========================
def find_latest_weights(weights_dir):
    weights = list(Path(weights_dir).glob("epoch_*.pt"))
    if not weights:
        latest_pt = Path(weights_dir) / "latest.pt"
        if latest_pt.exists(): return str(latest_pt)
        raise FileNotFoundError(f"No weights found in {weights_dir}")
    
    def get_epoch(p):
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    
    latest = max(weights, key=get_epoch)
    return str(latest)

# =========================
# Main
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    weights_path = find_latest_weights(WEIGHTS_DIR)
    print(f"Loading weights from: {weights_path}")

    # Initialize model
    netG = ResnetGenerator().to(device)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'netG' in checkpoint:
        netG.load_state_dict(checkpoint['netG'])
    else:
        netG.load_state_dict(checkpoint)
    
    netG.eval()

    if not os.path.exists(INPUT_IMG_PATH):
        raise FileNotFoundError(f"Image not found at {INPUT_IMG_PATH}")
        
    # === הלוגיקה החדשה לשמירה על רזולוציה ===
    
    # 1. טעינת התמונה המקורית
    img = Image.open(INPUT_IMG_PATH).convert("RGB")
    w, h = img.size
    
    # 2. חיתוך קל כדי שהמידות יתחלקו ב-4 (קריטי למודל)
    new_w = w - (w % 4)
    new_h = h - (h % 4)
    
    if new_w != w or new_h != h:
        print(f"[INFO] Cropping image slightly to fit model: {w}x{h} -> {new_w}x{new_h}")
        img = img.crop((0, 0, new_w, new_h))

    # 3. הגדרת הטרנספורמציה (בלי Resize!)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. יצירת הטנסור (השורה שהייתה חסרה לך)
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Run
    print("Running inference...")
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # Post-process & Save
    output_img = output_tensor.squeeze(0).cpu()
    output_img = (output_img * 0.5 + 0.5).clamp(0, 1)
    output_pil = transforms.ToPILImage()(output_img)
    output_pil.save(OUTPUT_PATH)
    
    print(f"✅ Success! Saved result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()