import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import re

# =========================
# הגדרות
# =========================
INPUT_IMG_PATH = "/home/nitzandu/synthetic-to-real-chessboard/temp_data/zoomed/game5/frame_1444/2_west.png"
WEIGHTS_DIR = "/home/nitzandu/synthetic-to-real-chessboard/outputs/cut_1_8X8/weights"
OUTPUT_PATH = "/home/nitzandu/synthetic-to-real-chessboard/data_test/full_cpu_result.png"

TARGET_SIZE = 2048  # הגודל המקורי שאת רוצה
BORDER_CUT = 0

# =========================
# המודל (אותו אחד בדיוק)
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
    if not weights: raise FileNotFoundError(f"No weights found in {weights_dir}")
    return str(max(weights, key=lambda p: int(re.search(r"epoch_(\d+)\.pt$", p.name).group(1))))

def main():
    # --- שינוי קריטי: מכריחים שימוש ב-CPU ---
    # ה-CPU איטי יותר אבל יש לו המון זיכרון (RAM) בניגוד ל-GPU
    device = "cpu" 
    print(f"Using device: {device} (This might take 1-2 minutes, but guarantees consistency)")

    # 1. טעינת מודל
    weights_path = find_latest_weights(WEIGHTS_DIR)
    netG = ResnetGenerator().to(device)
    
    # טעינת המשקולות ל-CPU
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['netG'] if 'netG' in ckpt else ckpt
    netG.load_state_dict(state_dict)
    netG.eval()

    # 2. הכנת תמונה
    img = Image.open(INPUT_IMG_PATH).convert("RGB")
    w, h = img.size
    
    if BORDER_CUT > 0:
        img = img.crop((BORDER_CUT, BORDER_CUT, w - BORDER_CUT, h - BORDER_CUT))

    # חשוב: Resize איכותי
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    print(f"[INFO] Processing full image size: {img.size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 3. הרצה (לוקח זמן!)
    print("[INFO] Running inference on full image... please wait...")
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # 4. שמירה
    output_img = output_tensor.squeeze(0).cpu()
    output_img = (output_img * 0.5 + 0.5).clamp(0, 1)
    final_pil = transforms.ToPILImage()(output_img)
    
    final_pil.save(OUTPUT_PATH)
    print(f"✅ Saved Full-Context Result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()