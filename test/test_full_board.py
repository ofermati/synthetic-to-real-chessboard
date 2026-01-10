import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
import re

# =========================
# הגדרות - לשנות כאן
# =========================
INPUT_IMG_PATH = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/synthetic/Game4/frame_616/G4_616_e_r6_c2.png"
WEIGHTS_DIR = "/home/nitzandu/synthetic-to-real-chessboard/outputs/cut_1_8X8/weights"
OUTPUT_PATH = "/home/nitzandu/synthetic-to-real-chessboard/data_test/cut_1_8X8_full_5.png"

# כמה פיקסלים לחתוך מכל צד (המסגרת החומה)?
# תנסי לשחק עם המספר הזה. לפי התמונה זה נראה בערך 40-50 פיקסלים
BORDER_CUT = 0

# הגודל הסופי של הלוח (חייב להיות כפולה של 256 כדי שהמודל יעבוד בול)
TARGET_SIZE = 2048
TILE_SIZE = 256

# =========================
# Model (אותו מודל)
# =========================
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c, affine=False), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c, affine=False)
        )
    def forward(self, x): return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=9):
        super().__init__()
        self.enc0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_c, ngf, 7, 1, 0), nn.InstanceNorm2d(ngf, affine=False), nn.ReLU(True))
        self.enc1 = nn.Sequential(nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2, affine=False), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4, affine=False), nn.ReLU(True))
        self.res = nn.Sequential(*[ResBlock(ngf*4) for _ in range(n_blocks)])
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf*2, affine=False), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf, affine=False), nn.ReLU(True))
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_c, 7, 1, 0), nn.Tanh())

    def forward(self, x):
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.res(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.out(x)

def find_latest_weights(weights_dir):
    weights = list(Path(weights_dir).glob("epoch_*.pt"))
    if not weights:
        latest_pt = Path(weights_dir) / "latest.pt"
        if latest_pt.exists(): return str(latest_pt)
        raise FileNotFoundError(f"No weights found in {weights_dir}")
    latest = max(weights, key=lambda p: int(re.search(r"epoch_(\d+)\.pt$", p.name).group(1)))
    return str(latest)

# =========================
# Main Logic
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. טעינת מודל
    weights_path = find_latest_weights(WEIGHTS_DIR)
    netG = ResnetGenerator().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['netG'] if 'netG' in ckpt else ckpt
    netG.load_state_dict(state_dict)
    netG.eval()

    # 2. הכנת התמונה (קריטי!)
    img = Image.open(INPUT_IMG_PATH).convert("RGB")
    w, h = img.size
    print(f"[INFO] Original size: {w}x{h}")
    
    # חיתוך המסגרת החומה
    if BORDER_CUT > 0:
        # (left, upper, right, lower)
        img = img.crop((BORDER_CUT, BORDER_CUT, w - BORDER_CUT, h - BORDER_CUT))
        print(f"[INFO] After cropping border: {img.size}")
    
    # שינוי גודל חזרה ל-2048x2048 כדי שהמשבצות יהיו בול 256
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    print(f"[INFO] Resized to target: {img.size}")

    # קנבס לתוצאה
    final_image = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("[INFO] Starting processing (No Overlap)...")
    
    # 3. לולאת חיתוך והדבקה
    # כאן אנחנו בטוחים שהתמונה היא 2048, אז 8x8 משבצות של 256 ייכנסו בול
    for y in range(0, TARGET_SIZE, TILE_SIZE):
        for x in range(0, TARGET_SIZE, TILE_SIZE):
            
            # חיתוך
            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            
            # המרה
            input_tensor = transform(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                output_tensor = netG(input_tensor)
            
            # שמירה
            output_tile = output_tensor.squeeze(0).cpu()
            output_tile = (output_tile * 0.5 + 0.5).clamp(0, 1)
            pil_tile = transforms.ToPILImage()(output_tile)
            
            # הדבקה
            final_image.paste(pil_tile, (x, y))

    final_image.save(OUTPUT_PATH)
    print(f"✅ Saved fixed image to: {OUTPUT_PATH}")
    print("💡 Tip: If grid is slightly off, adjust BORDER_CUT variable.")

if __name__ == "__main__":
    main()