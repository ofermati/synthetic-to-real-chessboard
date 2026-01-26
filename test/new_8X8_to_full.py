import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import re
import argparse  # <--- Added for command-line arguments
import sys

# =========================
# Model (exact same one)
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
    """
    Automatically finds the weight file with the highest epoch number (e.g. epoch_020.pt)
    """
    weights = list(Path(weights_dir).glob("epoch_*.pt"))
    if not weights: 
        raise FileNotFoundError(f"No weights found in {weights_dir}")
    # Sort by epoch number in filename
    return str(max(weights, key=lambda p: int(re.search(r"epoch_(\d+)\.pt$", p.name).group(1))))

def main():
    # --- Command-line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input zoomed image")
    parser.add_argument("--weights_dir", required=True, help="Directory containing model weights")
    parser.add_argument("--output", required=True, help="Path to save the result")
    parser.add_argument("--target_size", type=int, default=2048, help="Resize target")
    parser.add_argument("--border_cut", type=int, default=0, help="Optional crop")
    
    args = parser.parse_args()

    # Use CPU to avoid memory issues (as requested)
    device = "cpu" 
    print(f"[INFO] Using device: {device}")

    # 1. Load weights
    # Use helper to find latest weights in directory
    try:
        weights_path = find_latest_weights(args.weights_dir)
        print(f"[INFO] Loading weights from: {weights_path}")
    except Exception as e:
        print(f"[ERROR] Could not find weights: {e}")
        sys.exit(1)

    # Load model
    netG = ResnetGenerator().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['netG'] if 'netG' in ckpt else ckpt
    netG.load_state_dict(state_dict)
    netG.eval()

    # 2. Prepare image
    if not Path(args.input).exists():
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    
    if args.border_cut > 0:
        img = img.crop((args.border_cut, args.border_cut, w - args.border_cut, h - args.border_cut))

    # Resize
    img = img.resize((args.target_size, args.target_size), Image.BICUBIC)
    print(f"[INFO] Processing full image size: {img.size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Inference
    print("[INFO] Running inference on full image... please wait...")
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # 4. Save result
    output_img = output_tensor.squeeze(0).cpu()
    output_img = (output_img * 0.5 + 0.5).clamp(0, 1)
    final_pil = transforms.ToPILImage()(output_img)
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    final_pil.save(args.output)
    print(f"✅ Saved Full-Context Result to: {args.output}")

if __name__ == "__main__":
    main()