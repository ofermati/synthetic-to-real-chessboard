import os
import shutil
import tqdm

# --- הגדרות ---
# הנתיב לתיקיית ה-Synthetic
SOURCE_DIR = "/home/nitzandu/synthetic-to-real-chessboard/datasets/cut_8X8/synthetic"

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory not found: {SOURCE_DIR}")
        return

    # איסוף כל קבצי התמונה (png/jpg)
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg'))]
    
    print(f"Found {len(files)} synthetic patches. Organizing into Game/Frame folders...")

    count_moved = 0
    for filename in tqdm.tqdm(files):
        # ניתוח השם: G2_620_w_r0_c4.png
        parts = filename.split('_')
        
        # בדיקת תקינות השם (חייב להכיל לפחות משחק ופריים)
        if len(parts) < 2:
            continue
            
        game_tag = parts[0]  # G2
        frame_tag = parts[1] # 620
        
        # המרה לשמות תיקייה יפים
        game_folder_name = game_tag.replace("G", "Game") # Game2
        frame_folder_name = f"frame_{frame_tag}"         # frame_620
        
        # יצירת הנתיב המלא
        target_dir = os.path.join(SOURCE_DIR, game_folder_name, frame_folder_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # העברה
        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(target_dir, filename)
        
        shutil.move(src_path, dst_path)
        count_moved += 1

    print(f"Done! Organized {count_moved} files.")

if __name__ == "__main__":
    main()