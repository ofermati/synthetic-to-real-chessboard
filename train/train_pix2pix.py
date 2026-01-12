import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
import random

# =========================
# Config
# =========================
PROJECT_ROOT = "/home/nitzandu/synthetic-to-real-chessboard"
SYN_DIR = f"{PROJECT_ROOT}/datasets/cut_8X8/synthetic"
REAL_DIR = f"{PROJECT_ROOT}/datasets/cut_8X8/real"
RUN_DIR = f"{PROJECT_ROOT}/outputs/pix2pix_cutPictures"
IMG_OUT_DIR = f"{RUN_DIR}/images"
W_OUT_DIR = f"{RUN_DIR}/weights"

BATCH_SIZE = 4
EPOCHS = 100 
LR = 2e-4
LAMBDA_L1 = 100
IMG_SIZE = 256

# =========================
# Dataset מעודכן למבנה של Game -> Frame -> Images
# =========================
class PairedDataset(Dataset):
    def __init__(self, syn_root, real_root, img_size=256):
        self.samples = []
        self.img_size = img_size
        
        # 1. רצים על התיקיות של המשחקים (Game1, Game2...)
        games = sorted(os.listdir(syn_root))
        for game in games:
            syn_game_path = os.path.join(syn_root, game)
            real_game_path = os.path.join(real_root, game)
            
            # מוודאים שתיקיית המשחק קיימת בשני המקומות
            if os.path.isdir(syn_game_path) and os.path.isdir(real_game_path):
                
                # 2. רצים על תיקיות הפריימים (frame_0, frame_1...) בתוך המשחק
                frames = sorted(os.listdir(syn_game_path))
                for frame in frames:
                    syn_frame_path = os.path.join(syn_game_path, frame)
                    real_frame_path = os.path.join(real_game_path, frame)
                    
                    # מוודאים שתיקיית הפריים קיימת בשני המקומות
                    if os.path.isdir(syn_frame_path) and os.path.isdir(real_frame_path):
                        
                        # 3. רצים על התמונות החתוכות בתוך הפריים
                        images = sorted(os.listdir(syn_frame_path))
                        for img_name in images:
                            syn_img_final = os.path.join(syn_frame_path, img_name)
                            real_img_final = os.path.join(real_frame_path, img_name)
                            
                            # מוודאים שקובץ התמונה האמיתי קיים ושזה אכן קובץ
                            if os.path.isfile(syn_img_final) and os.path.isfile(real_img_final):
                                self.samples.append((syn_img_final, real_img_final))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        syn_path, real_path = self.samples[idx]
        syn = Image.open(syn_path).convert("RGB")
        real = Image.open(real_path).convert("RGB")

        # 1. Resize
        resize = transforms.Resize((self.img_size + 30, self.img_size + 30))
        syn, real = resize(syn), resize(real)

        # 2. Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(syn, output_size=(self.img_size, self.img_size))
        syn = TF.crop(syn, i, j, h, w)
        real = TF.crop(real, i, j, h, w)

        # 3. Random Horizontal Flip
        if random.random() > 0.5:
            syn = TF.hflip(syn)
            real = TF.hflip(real)

        # 4. ToTensor & Normalize
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        return tf(syn), tf(real)

# =========================
# Blocks
# =========================
class UNetBlockDown(nn.Module):
    def __init__(self, in_c, out_c, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

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

    def forward(self, x): return self.net(x)

# =========================
# Models
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

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, syn, tgt):
        return self.net(torch.cat([syn, tgt], dim=1))

# =========================
# Train Loop
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    os.makedirs(W_OUT_DIR, exist_ok=True)

    print(f"Loading dataset from: {SYN_DIR}...")
    dataset = PairedDataset(SYN_DIR, REAL_DIR)
    print(f"--> Found {len(dataset)} paired images for training.")
    
    if len(dataset) == 0:
        print("ERROR: No images found! Check paths.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    for epoch in range(EPOCHS):
        for i, (syn, real) in enumerate(loader):
            syn, real = syn.to(device), real.to(device)

            # Train D
            fake = G(syn).detach()
            loss_D = (criterion_GAN(D(syn, real), torch.ones_like(D(syn, real))) + 
                      criterion_GAN(D(syn, fake), torch.zeros_like(D(syn, fake)))) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Train G
            fake = G(syn)
            loss_G_GAN = criterion_GAN(D(syn, fake), torch.ones_like(D(syn, fake)))
            loss_G_L1 = criterion_L1(fake, real) * LAMBDA_L1
            loss_G = loss_G_GAN + loss_G_L1
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            if i % 20 == 0:
                print(f"E[{epoch}] Step[{i}/{len(loader)}] LossG: {loss_G.item():.3f} LossD: {loss_D.item():.3f}")

        # Save Preview
        with torch.no_grad():
            G.eval()
            test_syn, test_real = next(iter(loader))
            test_fake = G(test_syn.to(device))
            grid = torch.cat([test_syn[:2], test_fake.cpu()[:2], test_real[:2]], dim=0)
            save_image(grid, f"{IMG_OUT_DIR}/epoch_{epoch}.png", nrow=2, normalize=True)
            G.train()

        if epoch % 10 == 0:
            torch.save(G.state_dict(), f"{W_OUT_DIR}/G_{epoch}.pth")

if __name__ == "__main__":
    main()