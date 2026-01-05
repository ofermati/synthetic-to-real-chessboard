import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# =========================
# Core warp utils (based on your warp_board.py)
# =========================

def order_points(pts: np.ndarray) -> np.ndarray:
    """top-left, top-right, bottom-right, bottom-left"""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_board_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Find board as large 4-corner contour.
    Returns 4 points (x,y) or None.
    """
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
        if area < 20000:
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
    warped = cv2.warpPerspective(image_bgr, M, (out_size, out_size))
    return warped


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
# Pairing logic
# =========================

def to_game_capitalized(name: str) -> str:
    """
    Ensure 'game2' -> 'Game2', 'Game12' -> 'Game12'
    If already 'Game2' keep as-is.
    """
    s = name.strip()
    if not s:
        return s
    if s.lower().startswith("game"):
        return "Game" + s[4:]
    return s[0].upper() + s[1:]


def find_real_game_dir(real_base: Path, game_cap: str) -> Optional[Path]:
    """
    Try to find real directory as:
      real_base/Game2 OR real_base/game2
    """
    cand1 = real_base / game_cap
    cand2 = real_base / game_cap.lower()
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def main():
    PROJECT_ROOT = Path("/home/nitzandu/synthetic-to-real-chessboard")

    # Input synthetic renders (warped into paired)
    RENDERS_PAIRS = PROJECT_ROOT / "temp_data" / "renders_pairs"

    # Input real frames (already exist)
    REAL_UNPAIRED = PROJECT_ROOT / "datasets" / "unpaired" / "real"

    # Output paired
    PAIRED_OUT = PROJECT_ROOT / "datasets" / "paired"
    OUT_REAL = PAIRED_OUT / "real"
    OUT_SYN  = PAIRED_OUT / "synthetic"

    OUT_SIZE = 800  # change if needed

    if not RENDERS_PAIRS.exists():
        raise FileNotFoundError(f"Missing input renders_pairs: {RENDERS_PAIRS}")
    if not REAL_UNPAIRED.exists():
        raise FileNotFoundError(f"Missing real unpaired folder: {REAL_UNPAIRED}")

    OUT_REAL.mkdir(parents=True, exist_ok=True)
    OUT_SYN.mkdir(parents=True, exist_ok=True)

    # game2, game4, ...
    game_dirs = [
        d for d in RENDERS_PAIRS.iterdir()
        if d.is_dir() and d.name.lower().startswith("game")
    ]
    if not game_dirs:
        raise FileNotFoundError(f"No game folders under: {RENDERS_PAIRS}")

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("INPUT synthetic renders_pairs:", RENDERS_PAIRS)
    print("INPUT real unpaired:", REAL_UNPAIRED)
    print("OUTPUT paired:", PAIRED_OUT)
    print("OUT_SIZE:", OUT_SIZE)
    print(f"Found {len(game_dirs)} game folders")

    exts = {".png", ".jpg", ".jpeg"}

    for syn_game_dir in sorted(game_dirs):
        game_raw = syn_game_dir.name          # e.g. game2
        game_cap = to_game_capitalized(game_raw)  # -> Game2

        real_game_dir = find_real_game_dir(REAL_UNPAIRED, game_cap)
        if real_game_dir is None:
            print("\n" + "=" * 60)
            print(f"⚠️ SKIP GAME (no matching real folder): {game_cap}")
            print(f"Looked for: {REAL_UNPAIRED/game_cap} or {REAL_UNPAIRED/game_cap.lower()}")
            print("=" * 60)
            continue

        out_real_game = OUT_REAL / game_cap
        out_syn_game  = OUT_SYN  / game_cap
        out_real_game.mkdir(parents=True, exist_ok=True)
        out_syn_game.mkdir(parents=True, exist_ok=True)

        syn_images = [p for p in syn_game_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        syn_images.sort()

        print("\n" + "=" * 60)
        print(f"PROCESSING GAME: {game_cap}")
        print(f"SYN FROM: {syn_game_dir}")
        print(f"REAL FROM:{real_game_dir}")
        print(f"TO REAL:  {out_real_game}")
        print(f"TO SYN:   {out_syn_game}")
        print(f"Found {len(syn_images)} synthetic images")
        print("=" * 60)

        ok = skipped = missing_real = failed = 0

        for i, syn_path in enumerate(syn_images, start=1):
            frame_name = syn_path.name  # e.g. frame_000200.jpg

            real_path = real_game_dir / frame_name
            if not real_path.exists():
                missing_real += 1
                print(f"[{i}/{len(syn_images)}] ⚠️ missing real: {game_cap}/{frame_name}")
                continue

            out_syn_path  = out_syn_game / frame_name
            out_real_path = out_real_game / frame_name

            # skip if both exist already
            if out_syn_path.exists() and out_real_path.exists():
                skipped += 1
                continue

            try:
                # warp both so they match size and remove borders
                ok1 = process_one_image(syn_path, out_syn_path, out_size=OUT_SIZE)
                ok2 = process_one_image(real_path, out_real_path, out_size=OUT_SIZE)

                if ok1 and ok2:
                    ok += 1
                    if ok % 50 == 0:
                        print(f"✅ progress: {ok} pairs done in {game_cap}")
                else:
                    failed += 1
                    print(f"[{i}/{len(syn_images)}] ❌ failed pair: {game_cap}/{frame_name}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(syn_images)}] ❌ exception: {game_cap}/{frame_name} -> {e}")

        print("\n---- GAME SUMMARY ----")
        print("OK pairs:", ok)
        print("SKIPPED (already existed):", skipped)
        print("MISSING_REAL:", missing_real)
        print("FAILED:", failed)

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
