# inference_cyclegan_one_image.py
import torch
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.networks import NetConfig, build_generator




def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)


def main():
    # ======================
    # CONFIG - תעדכני כאן
    # ======================
    CKPT_PATH = Path("outputs/cyclegan_run1/G_S2R_epoch21.pth")  # המשקולות שלך
    INPUT_IMG = Path("datasets/cut_8X8/synthetic/Game7/frame_736/G7_736_e_r0_c0.png")                            # תמונת קלט לבדיקה
    OUTPUT_IMG = Path("output_cyclegan.png")                    # איפה לשמור פלט

    IMG_SIZE = (152, 152)                                       # חייב להתאים לאימון שלך

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not INPUT_IMG.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMG}")

    # ======================
    # Transform (כמו באימון)
    # ======================
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    img = Image.open(INPUT_IMG).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    # ======================
    # Build + load CycleGAN generator (S->R)
    # ======================
    cfg = NetConfig(img_channels=3, norm_g="instance", norm_d="instance", gan_mode="lsgan")
    G_S2R = build_generator("resnet", cfg, n_blocks=9).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    G_S2R.load_state_dict(state)
    G_S2R.eval()

    # ======================
    # Inference + save
    # ======================
    with torch.no_grad():
        y = G_S2R(x)  # [1,3,H,W]
        y = denorm(y).squeeze(0).cpu()

    to_pil = transforms.ToPILImage()
    out_img = to_pil(y)
    out_img.save(OUTPUT_IMG)

    print(f"Saved output to: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
