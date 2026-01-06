import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# =========================
# Core warp utils
# =========================

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_board_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)

        # במקום סף קבוע, סף יחסי לגודל התמונה (יותר יציב כשהלוח קטן)
        H, W = gray.shape[:2]
        min_area = 0.002 * H * W  # 0.2% מהתמונה (תשחקי: 0.001–0.01)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            return order_points(quad)

    return None

def warp_to_square(image_bgr: np.ndarray, quad: np.ndarray, out_size: int = 800) -> np.ndarray:
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr, M, (out_size, out_size))

def process_one_image(in_path: Path, out_path: Path, out_size: int = 800) -> bool:
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"❌ Can't read image: {in_path}")
        return False

    quad = find_board_quad(img)
    if quad is None:
        print(f"⚠️ Board quad not found: {in_path}")
        return False

    warped = warp_to_square(img, quad, out_size=out_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), warped)
    if not ok:
        print(f"❌ Failed saving: {out_path}")
    return ok

# =========================
# Main: warp ONLY synthetic renders_pairs
# =========================

def to_game_capitalized(name: str) -> str:
    s = name.strip()
    if s.lower().startswith("game"):
        return "Game" + s[4:]
    return s[0].upper() + s[1:]

def main():
    PROJECT_ROOT = Path("/home/nitzandu/synthetic-to-real-chessboard")

    IN_SYN = PROJECT_ROOT / "temp_data" / "renders_pairs"
    OUT_SYN = PROJECT_ROOT / "datasets" / "paired" / "synthetic"

    OUT_SIZE = 800
    exts = {".png", ".jpg", ".jpeg"}

    if not IN_SYN.exists():
        raise FileNotFoundError(f"Missing input: {IN_SYN}")

    game_dirs = [d for d in IN_SYN.iterdir() if d.is_dir() and d.name.lower().startswith("game")]
    if not game_dirs:
        raise FileNotFoundError(f"No game folders under: {IN_SYN}")

    print("INPUT:", IN_SYN)
    print("OUTPUT:", OUT_SYN)
    print("OUT_SIZE:", OUT_SIZE)
    print(f"Found {len(game_dirs)} game folders")

    total_ok = total_fail = total_skip = 0

    for syn_game_dir in sorted(game_dirs):
        game_cap = to_game_capitalized(syn_game_dir.name)
        out_game = OUT_SYN / game_cap
        out_game.mkdir(parents=True, exist_ok=True)

        syn_images = [p for p in syn_game_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        syn_images.sort()

        print("\n" + "=" * 60)
        print(f"GAME: {game_cap} | images: {len(syn_images)}")
        print("=" * 60)

        ok = fail = skip = 0
        for i, syn_path in enumerate(syn_images, start=1):
            out_path = out_game / syn_path.name

            if out_path.exists():
                skip += 1
                continue

            if process_one_image(syn_path, out_path, out_size=OUT_SIZE):
                ok += 1
                if ok % 50 == 0:
                    print(f"✅ progress: {ok} saved in {game_cap}")
            else:
                fail += 1
                print(f"[{i}/{len(syn_images)}] ❌ failed: {game_cap}/{syn_path.name}")

        print(f"SUMMARY {game_cap}: OK={ok} SKIP={skip} FAIL={fail}")
        total_ok += ok
        total_fail += fail
        total_skip += skip

    print("\nALL DONE.")
    print(f"TOTAL: OK={total_ok} SKIP={total_skip} FAIL={total_fail}")

if __name__ == "__main__":
    main()
