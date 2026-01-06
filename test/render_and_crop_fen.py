import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


# =========================
# Paths (קבועים אצלך)
# =========================
BLENDER_BIN = "/home/nitzandu/blender-3.6.5-linux-x64/blender"
BLEND_FILE = "/home/nitzandu/synthetic-to-real-chessboard/blender/chess-set.blend"
BLENDER_SCRIPT = "/home/nitzandu/synthetic-to-real-chessboard/blender/chess_position_api_v2.py"

OUT_DIR = "/home/nitzandu/synthetic-to-real-chessboard/data_test"
TMP_DIR = os.path.join(OUT_DIR, "_tmp")

RENDER_RESOLUTION = 1024
CAMERA_VIEW = "black"   # כמו בסקריפט שלך
OVERHEAD_FILENAME = "1_overhead.png"


# =========================
# Crop helpers
# =========================
def estimate_background_color(img_np: np.ndarray) -> np.ndarray:
    """Estimate background color using the 4 image corners."""
    h, w, _ = img_np.shape
    patch = 20
    corners = [
        img_np[0:patch, 0:patch],
        img_np[0:patch, w - patch:w],
        img_np[h - patch:h, 0:patch],
        img_np[h - patch:h, w - patch:w],
    ]
    return np.mean(
        np.concatenate([c.reshape(-1, 3) for c in corners], axis=0),
        axis=0
    )


def auto_crop_board(
    in_path: str,
    out_path: str,
    threshold: float = 18.0,
    margin: int = 12
):
    """Automatically crop chessboard from gray background."""
    img = Image.open(in_path).convert("RGB")
    img_np = np.array(img)

    bg = estimate_background_color(img_np)
    diff = np.linalg.norm(img_np.astype(np.float32) - bg, axis=2)
    mask = diff > threshold

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        img.save(out_path)
        return

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    h, w = img_np.shape[:2]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h - 1, y1 + margin)
    x1 = min(w - 1, x1 + margin)

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    cropped.save(out_path)


# =========================
# Main pipeline
# =========================
def main():
    # ---- Ask user for FEN ----
    print("\nPlease enter FEN (press Enter when done):")
    fen = input().strip()

    if not fen:
        raise RuntimeError("No FEN provided.")

    # Blender script expects only the board part
    fen_board_only = fen.split()[0]

    # ---- Prepare folders ----
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # ---- Run Blender (render ONE image set) ----
    cmd = [
        BLENDER_BIN,
        BLEND_FILE,
        "--background",
        "--python",
        BLENDER_SCRIPT,
        "--",
        "--fen",
        fen_board_only,
        "--view",
        CAMERA_VIEW,
        "--resolution",
        str(RENDER_RESOLUTION),
        "--output_dir",
        TMP_DIR,
    ]

    print("\nRunning Blender...")
    subprocess.run(cmd, check=True)

    overhead_path = os.path.join(TMP_DIR, OVERHEAD_FILENAME)
    if not os.path.exists(overhead_path):
        raise RuntimeError("Overhead image was not created.")

    # ---- Crop ----
    final_path = os.path.join(OUT_DIR, "synthetic_from_fen.png")
    auto_crop_board(overhead_path, final_path)

    # ---- Cleanup tmp ----
    try:
        for p in Path(TMP_DIR).glob("*"):
            p.unlink()
        Path(TMP_DIR).rmdir()
    except Exception:
        pass

    print("\nDONE ✅")
    print("Saved cropped image to:")
    print(final_path)


if __name__ == "__main__":
    main()
