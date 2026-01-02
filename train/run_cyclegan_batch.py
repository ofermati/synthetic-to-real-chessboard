import os
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from models.networks import NetConfig, build_generator
from tqdm import tqdm  # progress bar

# ===========================
# CONFIG
# ===========================
INPUT_ROOT = Path("renders")           # Where blender output is
OUTPUT_ROOT = Path("renders_processed") # Where processed images go
IMAGE_SIZE = (256, 256)                # Resize before processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_cyclegan_model():
    """Builds and returns the CycleGAN generator."""
    print(f"🔧 Building CycleGAN Generator (ResNet) on {DEVICE}...")
    
    # Same config as used in test_single_image.py for CycleGAN
    cfg_cycle = NetConfig(img_channels=3, norm_g="instance", norm_d="instance", gan_mode="lsgan")
    model = build_generator("resnet", cfg_cycle, n_blocks=9)
    
    model.to(DEVICE)
    model.eval()  # Set to inference mode
    return model

def process_image(model, img_path, save_path, transform):
    """Loads image, runs model, saves output."""
    try:
        # Load
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(DEVICE) # (1, 3, H, W)
        
        # Inference
        with torch.no_grad():
            fake_img_tensor = model(img_tensor)
            
        # Postprocess (from [-1, 1] to [0, 1])
        fake_img_tensor = (fake_img_tensor + 1) / 2.0
        
        # Save
        to_pil = transforms.ToPILImage()
        result_img = to_pil(fake_img_tensor.squeeze(0).cpu())
        
        # Create parent dir if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result_img.save(save_path)
        
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")

def main():
    if not INPUT_ROOT.exists():
        print(f"❌ Input folder '{INPUT_ROOT}' does not exist.")
        return

    # 1. Load Model
    generator = load_cyclegan_model()

    # 2. Define Transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 3. Find all PNG images recursively
    print(f"📂 Scanning {INPUT_ROOT} for images...")
    all_images = list(INPUT_ROOT.rglob("*.png"))
    
    if not all_images:
        print("⚠️ No images found!")
        return
        
    print(f"found {len(all_images)} images. Starting processing...")

    # 4. Process Loop
    for img_path in tqdm(all_images):
        # Construct output path: renders/game2/frame.../img.png -> renders_processed/game2/frame.../img.png
        relative_path = img_path.relative_to(INPUT_ROOT)
        save_path = OUTPUT_ROOT / relative_path
        
        process_image(generator, img_path, save_path, transform)

    print("\n✨ Done!")
    print(f"💾 Processed images saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()

