import os
import cv2
import glob
from pathlib import Path
import tqdm

# --- הגדרות ---
SOURCE_ROOT = "/home/nitzandu/synthetic-to-real-chessboard/temp_data/zoomed"
OUTPUT_DIR = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/synthetic"
TARGET_SIZE = (800, 800)
PADDING_RATIO = 0.25 

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_view_char(view_string):
    """מחזיר תו בודד המייצג את הזווית"""
    view_lower = view_string.lower()
    if "west" in view_lower:
        return "w"
    elif "east" in view_lower:
        return "e"
    elif "overhead" in view_lower:
        return "o"
    else:
        return view_lower[0] # ברירת מחדל: האות הראשונה

def process_and_save(img_path, output_dir):
    path_obj = Path(img_path)
    
    # חילוץ שמות מתוך הנתיב
    # מבנה: .../game2/frame_620/2_west.png
    view_raw = path_obj.stem       # 2_west
    frame_dir = path_obj.parent.name # frame_620
    game_dir = path_obj.parent.parent.name # game2
    
    if "game" not in game_dir or "frame" not in frame_dir:
        return # דילוג על קבצים לא רלוונטיים

    # --- יצירת השם הקצר ---
    game_num = game_dir.replace("game", "")      # "2"
    frame_num = frame_dir.replace("frame_", "")  # "620"
    view_char = get_view_char(view_raw)          # "w"
    
    # טעינת התמונה
    img = cv2.imread(str(img_path))
    if img is None:
        return

    # Resize & Padding
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    h, w = img_resized.shape[:2]
    square_h = h // 8
    square_w = w // 8
    pad_h = int(square_h * PADDING_RATIO)
    pad_w = int(square_w * PADDING_RATIO)
    
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101
    )
    
    # חיתוך ושמירה
    for row in range(8):
        for col in range(8):
            y1 = row * square_h
            y2 = y1 + square_h + (2 * pad_h)
            x1 = col * square_w
            x2 = x1 + square_w + (2 * pad_w)
            
            patch = img_padded[y1:y2, x1:x2]
            
            # --- כאן נוצר השם החדש ---
            # פורמט: G2_620_w_r0_c0.png
            save_name = f"G{game_num}_{frame_num}_{view_char}_r{row}_c{col}.png"
            
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, patch)

def main():
    ensure_dir(OUTPUT_DIR)
    
    # ניקוי התיקייה לפני ריצה חדשה (כדי שלא יהיו כפילויות עם השמות הישנים)
    print("Cleaning old files in output directory...")
    old_files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
    for f in old_files:
        os.remove(f)

    search_pattern = os.path.join(SOURCE_ROOT, "game*", "frame_*", "*.png")
    files = glob.glob(search_pattern)
    
    print(f"Found {len(files)} synthetic images. Processing...")
    
    for file_path in tqdm.tqdm(files):
        process_and_save(file_path, OUTPUT_DIR)
        
    print(f"Done! Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()