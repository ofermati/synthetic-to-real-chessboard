import os
import sys
import shutil
import subprocess
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# =========================
# 1) הגדרות (Configuration)
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# נתיבים
WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "cut_1_8X8_new" / "weights" / "latest.pt"
BLEND_FILE = PROJECT_ROOT / "blender" / "chess-set.blend" 
# משתמשים בסקריפט ה-Batch שלך
BLENDER_SCRIPT = PROJECT_ROOT / "blender" / "chess_position_api_v2_cropped_batch.py"
# נתיב יציאה של בלנדר
BLENDER_OUTPUT_DIR = (PROJECT_ROOT / "blender" / "renders").resolve()

# === הגדרות קריטיות לאיכות ===
TARGET_SIZE = 2048  # הרזולוציה הגבוהה שביקשת
TILES_PER_SIDE = 8
BORDER_CUT = 0      # לפי הקובץ new_8X8_to_full.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) המודל (ResnetGenerator)
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

# =========================
# 3) פונקציות הליבה
# =========================

def find_blender_exe(project_root):
    # חיפוש בלנדר בתיקייה או ב-PATH
    search_paths = [project_root / "blender", project_root.parent]
    for root_dir in search_paths:
        if not root_dir.exists(): continue
        for path in root_dir.rglob("blender"):
            if path.is_file() and os.access(path, os.X_OK) and path.name == "blender": 
                return path
    system_blender = shutil.which("blender")
    if system_blender: return Path(system_blender)
    raise FileNotFoundError("Could not find 'blender' executable.")

def load_model(weights_path, device):
    if not weights_path.exists(): raise FileNotFoundError(f"Weights not found: {weights_path}")
    netG = ResnetGenerator().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['netG'] if 'netG' in ckpt else ckpt
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

def run_blender_generation(blender_exe, fen):
    """
    מריץ את בלנדר כדי לייצר תמונה סינתטית.
    התיקון החשוב: --resolution 2048 וניקוי קבצים קודמים.
    """
    print(f"[INFO] 1. Rendering synthetic image in Blender (2048x2048)...")
    
    BLENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # נשתמש בתמונה 1_overhead.png כברירת מחדל
    expected_output = BLENDER_OUTPUT_DIR / "1_overhead.png"

    # מחיקת קובץ ישן כדי למנוע מבלנדר לדלג עליו
    if expected_output.exists():
        try: os.remove(expected_output)
        except: pass

    # הפקודה לבלנדר
    cmd = [
        str(blender_exe), str(BLEND_FILE),
        "--background",
        "--python", str(BLENDER_SCRIPT),
        "--",
        "--fen", fen,
        "--resolution", str(TARGET_SIZE),  # <--- התיקון הקריטי: 2048
        "--output_dir", str(BLENDER_OUTPUT_DIR)
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if not expected_output.exists():
        print("[ERROR] Blender failed.")
        print(result.stderr)
        # הדפסת לוג אם צריך
        if "Error" in result.stdout:
            print(result.stdout[-500:])
        raise FileNotFoundError("Blender output not found")
        
    print(f"[SUCCESS] Synthetic image created at: {expected_output}")
    return expected_output

def process_image_full_logic(netG, img_path, device):
    """
    הלוגיקה המדויקת מהקובץ new_8X8_to_full.py:
    טעינה -> Resize (2048) -> חיתוך ל-8X8 -> הרצה -> הדבקה
    """
    print(f"[INFO] 2. Processing: Cut to 8x8 -> Inference -> Stitch...")
    
    # 1. הכנת תמונה
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    # חיתוך שוליים (אם הוגדר)
    if BORDER_CUT > 0:
        img = img.crop((BORDER_CUT, BORDER_CUT, w - BORDER_CUT, h - BORDER_CUT))

    # Resize איכותי ל-2048
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    
    # הכנת קנבס לתמונה הסופית
    final_result = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE))
    tile_size = TARGET_SIZE // TILES_PER_SIDE # 256
    
    # טרנספורמציה למודל
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("[INFO] Running inference on tiles...", end=" ")
    
    with torch.no_grad():
        for row in range(TILES_PER_SIDE):
            for col in range(TILES_PER_SIDE):
                # חישוב קואורדינטות
                left = col * tile_size
                upper = row * tile_size
                right = left + tile_size
                lower = upper + tile_size
                
                # חיתוך הריבוע (Crop)
                tile = img.crop((left, upper, right, lower))
                
                # הרצה במודל
                input_tensor = transform(tile).unsqueeze(0).to(device)
                fake_tile_tensor = netG(input_tensor)
                
                # המרה חזרה לתמונה
                fake_tile = fake_tile_tensor.squeeze(0).cpu()
                fake_tile = (fake_tile * 0.5 + 0.5).clamp(0, 1)
                tile_pil = transforms.ToPILImage()(fake_tile)
                
                # הדבקה לתמונה הגדולה (Paste)
                final_result.paste(tile_pil, (left, upper))
                print(".", end="", flush=True)

    print("\n[INFO] Stitching complete.")
    return final_result

# =========================
# 4) Main Loop
# =========================
def main():
    print("="*60)
    print(f"High-Res Pipeline: FEN -> Blender(2048) -> Split/Merge -> Result")
    print("="*60)

    # 1. הגדרות ראשוניות
    try:
        blender_exe = find_blender_exe(PROJECT_ROOT)
        if not BLENDER_SCRIPT.exists(): raise FileNotFoundError(f"Missing script: {BLENDER_SCRIPT}")
        if not BLEND_FILE.exists(): raise FileNotFoundError(f"Missing blend file: {BLEND_FILE}")
        netG = load_model(WEIGHTS_PATH, DEVICE)
    except Exception as e:
        print(f"[ERROR] Init failed: {e}")
        return

    # 2. לולאת עבודה
    while True:
        print("\n" + "-"*40)
        fen_input = input("Enter FEN (or 'q' to quit): ").strip()
        
        if fen_input.lower() == 'q':
            break
        if not fen_input:
            continue
            
        try:
            # שלב א': בלנדר (2048px)
            synth_path = run_blender_generation(blender_exe, fen_input)
            
            # שלב ב': חיתוך, מודל וחיבור
            final_image = process_image_full_logic(netG, synth_path, DEVICE)
            
            # שלב ג': שמירה
            output_filename = f"final_hq_{fen_input[:5].replace('/', '_')}.png"
            output_path = SCRIPT_DIR / output_filename
            final_image.save(output_path)
            
            print(f"✅ FINAL IMAGE SAVED: {output_path}")
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()