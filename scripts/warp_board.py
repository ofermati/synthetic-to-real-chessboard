import cv2
import numpy as np
from pathlib import Path

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    מסדרת 4 נקודות לפי סדר קבוע:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)          # x+y
    diff = np.diff(pts, axis=1)  # x-y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_board_quad(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    מנסה למצוא את הלוח בתמונה בתור קונטור גדול עם 4 פינות.
    מחזירה 4 נקודות (x,y) אם נמצא, אחרת None.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # נבדוק קונטורים מהגדול לקטן
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < 20000:  # סף כדי לא לבחור רעשים קטנים (תשני אם צריך)
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # רוצים צורה עם 4 נקודות
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            return order_points(quad)

    return None

def warp_to_square(image_bgr: np.ndarray, quad: np.ndarray, out_size: int = 800) -> np.ndarray:
    """
    מבצעת Perspective Warp כדי להפוך את ה-quad לריבוע out_size x out_size.
    """
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_size, out_size))
    return warped

def process_one_image(in_path: Path, out_path: Path, out_size: int = 800) -> bool:
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"❌ Can't read image: {in_path}")
        return False

    quad = find_board_quad(img)
    if quad is None:
        print(f"⚠️ Board not found (quad). Try changing thresholds: {in_path}")
        return False

    warped = warp_to_square(img, quad, out_size=out_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), warped)
    if ok:
        print(f"✅ Saved: {out_path}")
    else:
        print(f"❌ Failed saving: {out_path}")
    return ok

def process_folder(in_root: Path, out_root: Path, out_size: int = 800) -> None:
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in in_root.rglob("*") if p.suffix.lower() in exts]

    print(f"Found {len(images)} images under {in_root}")
    success = 0

    for p in images:
        rel = p.relative_to(in_root)
        out_path = out_root / rel
        if process_one_image(p, out_path, out_size=out_size):
            success += 1

    print(f"\nDone. Success: {success}/{len(images)}")

if __name__ == "__main__":
    renders_base = Path(r"C:\Dl_project\renders")
    processed_base = Path(r"C:\Dl_project\processed")

    # כל תיקייה בתוך renders (Game4, Game5, ...)
    game_dirs = [d for d in renders_base.iterdir() if d.is_dir()]
    print(f"Found {len(game_dirs)} game folders under {renders_base}")

    for game_dir in sorted(game_dirs):
        out_dir = processed_base / game_dir.name
        print("\n" + "="*60)
        print(f"PROCESSING GAME FOLDER: {game_dir}")
        print(f"OUTPUT FOLDER:         {out_dir}")
        print("="*60)

        process_folder(game_dir, out_dir, out_size=800)
