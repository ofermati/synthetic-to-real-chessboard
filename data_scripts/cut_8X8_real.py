import os
import cv2
import glob
from pathlib import Path
import tqdm

# --- הגדרות ---
SOURCE_ROOT = "/home/nitzandu/synthetic-to-real-chessboard/datasets/unpaired/real"
OUTPUT_DIR = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/real"

TARGET_SIZE = (800, 800)
PADDING_RATIO = 0.25 

# מיפוי מספר משחק לזווית (כדי שיהיה תואם לסינתטי)
GAME_VIEW_MAP = {
    '2': 'w',
    '4': 'e',
    '5': 'w',
    '6': 'o',
    '7': 'e'
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def auto_crop_board(img):
    """חיתוך חכם להסרת שוליים שחורים"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img 
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def process_and_save(img_path, output_dir):
    path_obj = Path(img_path)
    filename = path_obj.stem       
    parent_dir = path_obj.parent.name 
    
    # חילוץ מידע מהשם
    game_num = parent_dir.lower().replace("game", "")
    frame_num = filename.lower().replace("frame_", "").lstrip("0")
    if frame_num == "": frame_num = "0"
    
    view_char = GAME_VIEW_MAP.get(game_num, 'u') 
    
    # טעינה ועיבוד
    img = cv2.imread(str(img_path))
    if img is None:
        return

    img_cropped = auto_crop_board(img)
    img_resized = cv2.resize(img_cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    h, w = img_resized.shape[:2]
    square_h = h // 8
    square_w = w // 8
    pad_h = int(square_h * PADDING_RATIO)
    pad_w = int(square_w * PADDING_RATIO)
    
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101
    )
    
    for row in range(8):
        for col in range(8):
            y1 = row * square_h
            y2 = y1 + square_h + (2 * pad_h)
            x1 = col * square_w
            x2 = x1 + square_w + (2 * pad_w)
            
            patch = img_padded[y1:y2, x1:x2]
            
            # --- הפורמט שרצית (עם r ו-c) ---
            # G2_200_w_r0_c4.png
            save_name = f"G{game_num}_{frame_num}_{view_char}_r{row}_c{col}.png"
            
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, patch)

def main():
    ensure_dir(OUTPUT_DIR)
    
    # מחיקת קבצים ישנים בתיקיית Real
    print("Cleaning old real files...")
    old_files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
    for f in old_files:
        os.remove(f)

    # חיפוש קבצים (תומך JPG ו-PNG)
    search_pattern = os.path.join(SOURCE_ROOT, "**", "*.jpg") 
    files = glob.glob(search_pattern, recursive=True)
    files.extend(glob.glob(os.path.join(SOURCE_ROOT, "**", "*.png"), recursive=True))
    
    print(f"Found {len(files)} real images. Processing to format: Gx_xxx_x_rX_cX...")
    
    for file_path in tqdm.tqdm(files):
        process_and_save(file_path, OUTPUT_DIR)
        
    print(f"Done! Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()