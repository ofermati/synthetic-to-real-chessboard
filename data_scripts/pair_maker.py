import re
import random
from pathlib import Path

import cv2
import numpy as np

# =========================
# הגדרות (לשנות רק כאן אם צריך)
# =========================
PROJECT_DIR = Path(r"C:\Dl_project")

PROCESSED_DIR = PROJECT_DIR / "processed"
REAL_DIR = PROJECT_DIR / "real_images"
OUT_DIR = PROJECT_DIR / "dataset_pix2pix"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# גודל לשמירה בדאטהסט (אימון pix2pix אוהב גדלים קבועים)
SAVE_SIZE = (512, 512)

# גודל להשוואת ECC (קטן = מהיר)
ECC_SIZE = (256, 256)
ECC_ITERS = 80  # אפשר 70-120; יותר = יציב יותר אבל איטי יותר

# להרצה ניסיונית:
MAX_PAIRS = None  # למשל 30 לבדיקה, או None להכול

FRAME_RE = re.compile(r"(frame[_-]?\d+)", re.IGNORECASE)


# =========================
# עזרי מערכת קבצים
# =========================
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_frame_id(path: Path) -> str | None:
    m = FRAME_RE.search(str(path))
    if not m:
        return None
    return m.group(1).lower().replace("-", "_")


def ensure_dirs(out_dir: Path):
    for split in ["train", "val", "test"]:
        for side in ["A", "B"]:
            (out_dir / f"{split}{side}").mkdir(parents=True, exist_ok=True)


def choose_split(items):
    random.seed(RANDOM_SEED)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }


# =========================
# מציאת real אצלך:
# real_images/GameX/images/frame_000028.png
# =========================
def find_real_image(real_root: Path, game_name: str, frame_id: str) -> Path | None:
    game_dir = real_root / game_name
    if not game_dir.exists():
        return None

    for ext in [".png", ".jpg", ".jpeg"]:
        p = game_dir / f"{frame_id}{ext}"
        if p.exists():
            return p

    return None


def load_bgr(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


# =========================
# ECC: לבחור את המועמד הכי "ניתן ליישור" ל-real
# =========================
def _prep_for_ecc(img_bgr, size=ECC_SIZE):
    """
    מכין תמונה ל-ECC בצורה עמידה לתאורה וצבע:
    grayscale + CLAHE + magnitude של gradient.
    """
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    mag = mag / (mag.max() + 1e-6)
    return mag.astype(np.float32)


def ecc_score(real_bgr, cand_bgr, iters=ECC_ITERS):
    """
    מחזיר ציון ECC (גבוה יותר = התאמה טובה יותר).
    משתמש ב-Affine כדי לפתור נטייה שמאלה/ימינה מהר.
    """
    template = _prep_for_ecc(real_bgr)
    image = _prep_for_ecc(cand_bgr)

    warp = np.eye(2, 3, dtype=np.float32)  # affine 2x3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5)

    try:
        cc, _ = cv2.findTransformECC(
            template, image, warp,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5
        )
        return float(cc)
    except cv2.error:
        return -1e9


def choose_best_by_ecc(real_bgr, syn_paths):
    best_p = None
    best_cc = -1e9

    for p in syn_paths:
        cand = load_bgr(p)
        if cand is None:
            continue
        cc = ecc_score(real_bgr, cand)
        if cc > best_cc:
            best_cc = cc
            best_p = p

    return best_p, best_cc


# =========================
# שמירת זוגות
# =========================
def save_pair(out_dir: Path, split: str, base_name: str, syn_path: Path, real_path: Path):
    out_a = out_dir / f"{split}A" / f"{base_name}.png"
    out_b = out_dir / f"{split}B" / f"{base_name}.png"

    syn = load_bgr(syn_path)
    real = load_bgr(real_path)

    if syn is None:
        raise ValueError(f"Failed to read synthetic: {syn_path}")
    if real is None:
        raise ValueError(f"Failed to read real: {real_path}")

    syn = cv2.resize(syn, SAVE_SIZE, interpolation=cv2.INTER_AREA)
    real = cv2.resize(real, SAVE_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(out_a), syn)
    cv2.imwrite(str(out_b), real)


# =========================
# MAIN
# =========================
def build_dataset():
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Processed dir not found: {PROCESSED_DIR}")
    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Real dir not found: {REAL_DIR}")

    ensure_dirs(OUT_DIR)

    pairs = []
    skipped = 0
    collected = 0

    # processed/GameX/frame_YYYYYY/
    for game_dir in PROCESSED_DIR.iterdir():
        if not game_dir.is_dir():
            continue

        game_name = game_dir.name  # Game1, Game2...

        for frame_dir in game_dir.iterdir():
            if not frame_dir.is_dir():
                continue

            frame_id = find_frame_id(frame_dir)
            if frame_id is None:
                continue

            real_path = find_real_image(REAL_DIR, game_name, frame_id)
            if real_path is None:
                print(f"[SKIP] no real for {game_name}/{frame_id}")
                skipped += 1
                continue

            syn_candidates = [p for p in frame_dir.iterdir() if p.is_file() and is_image(p)]
            if not syn_candidates:
                print(f"[SKIP] no syn images in {frame_dir}")
                skipped += 1
                continue

            real_bgr = load_bgr(real_path)
            if real_bgr is None:
                print(f"[SKIP] failed reading real {real_path}")
                skipped += 1
                continue

            best_syn, best_cc = choose_best_by_ecc(real_bgr, syn_candidates)
            if best_syn is None:
                print(f"[SKIP] could not choose view for {game_name}/{frame_id}")
                skipped += 1
                continue

            base_name = f"{game_name}_{frame_id}"
            pairs.append((base_name, best_syn, real_path))

            print(f"[PAIR] {base_name} best={best_syn.name} ecc={best_cc:.5f}")

            collected += 1
            if MAX_PAIRS is not None and collected >= MAX_PAIRS:
                break

        if MAX_PAIRS is not None and collected >= MAX_PAIRS:
            break

    if not pairs:
        print("No pairs found. Check folder structure.")
        return

    splits = choose_split(pairs)
    for split, items in splits.items():
        for base_name, syn_path, real_path in items:
            save_pair(OUT_DIR, split, base_name, syn_path, real_path)

    print("\nDONE!")
    print(f"Total pairs: {len(pairs)}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    build_dataset()
