import torch
from torchvision import transforms
from PIL import Image
from models.networks import NetConfig, build_generator

def run_inference(model_name, generator, img_tensor, output_filename):
    print(f"\n⚡ Running {model_name} model...")
    with torch.no_grad():
        fake_img_tensor = generator(img_tensor)
    
    print(f"✅ {model_name} output shape: {fake_img_tensor.shape}")

    # Convert back to [0, 1] range and save
    fake_img_tensor = (fake_img_tensor + 1) / 2.0 
    to_pil = transforms.ToPILImage()
    result_img = to_pil(fake_img_tensor.squeeze(0))
    
    result_img.save(output_filename)
    print(f"💾 Saved {model_name} output to: {output_filename}")

def test_run():
    print("🚀 Starting Multi-Model Test...")
    
    # 1. Setup paths and image
    syn_path = "data/syn.png"
    print(f"📂 Loading image from: {syn_path}")
    
    try:
        img = Image.open(syn_path).convert('RGB')
    except FileNotFoundError:
        print("❌ Error: Could not find data/syn.png")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0) # (1, 3, 256, 256)

    # ---------------------------
    # 2. Test Pix2Pix (U-Net)
    # ---------------------------
    print("\n🔧 Building Pix2Pix Generator (U-Net)...")
    cfg_p2p = NetConfig(img_channels=3, norm_g="batch", norm_d="batch", gan_mode="bce")
    G_pix2pix = build_generator("unet", cfg_p2p, num_downs=8)
    
    run_inference("Pix2Pix", G_pix2pix, img_tensor, "output_pix2pix.png")

    # ---------------------------
    # 3. Test CycleGAN (ResNet)
    # ---------------------------
    print("\n🔧 Building CycleGAN Generator (ResNet)...")
    cfg_cycle = NetConfig(img_channels=3, norm_g="instance", norm_d="instance", gan_mode="lsgan")
    G_cyclegan = build_generator("resnet", cfg_cycle, n_blocks=9)
    
    run_inference("CycleGAN", G_cyclegan, img_tensor, "output_cyclegan.png")

    print("\n✨ Done! Check the output files.")

if __name__ == "__main__":
    test_run()
