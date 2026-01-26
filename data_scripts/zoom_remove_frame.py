#!/usr/bin/env python3
"""
Zoom-crop all images under a root folder to remove a thin border/frame,
then resize back to the original size and write to an output root folder,
preserving relative paths.

Example:
    python /home/nitzandu/synthetic-to-real-chessboard/data_scripts/zoom_remove_frame.py \
        --in-root  /home/nitzandu/synthetic-to-real-chessboard/renders \
        --out-root /home/nitzandu/synthetic-to-real-chessboard/temp_data/zoomed \
        --zoom 0.90
"""

from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def center_crop_box(w: int, h: int, zoom: float) -> tuple[int, int, int, int]:
    """
    zoom < 1.0 means crop inwards (remove borders).
    Example zoom=0.97 keeps 97% of width/height.
    """
    if not (0.0 < zoom <= 1.0):
        raise ValueError("zoom must be in (0, 1].")

    new_w = max(1, int(round(w * zoom)))
    new_h = max(1, int(round(h * zoom)))

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    return left, top, right, bottom


def process_image(in_path: Path, out_path: Path, zoom: float, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as im:
        im = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im

        w, h = im.size
        box = center_crop_box(w, h, zoom)
        cropped = im.crop(box)

        # Resize back to original size (keeps final dataset resolution identical)
        resized = cropped.resize((w, h), resample=Image.Resampling.LANCZOS)

        # Keep PNG if input was PNG, otherwise keep original suffix
        suffix = in_path.suffix.lower()
        if suffix == ".png":
            resized.save(out_path.with_suffix(".png"), format="PNG", optimize=True)
        else:
            # For jpg/jpeg etc.
            resized.save(out_path, quality=95, optimize=True)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--zoom", type=float, default=0.97,
                    help="Keep fraction of image (e.g., 0.97 removes 1.5%% per side).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_root: Path = args.in_root
    out_root: Path = args.out_root
    zoom: float = args.zoom

    if not in_root.exists():
        raise SystemExit(f"[ERROR] in-root does not exist: {in_root}")

    files = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        print(f"[WARN] No images found under: {in_root}")
        return

    wrote = 0
    skipped = 0
    errors = 0

    for in_path in files:
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel

        if args.dry_run:
            print(f"[DRY] {in_path} -> {out_path} (zoom={zoom})")
            continue

        try:
            changed = process_image(in_path, out_path, zoom=zoom, overwrite=args.overwrite)
            if changed:
                wrote += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] {in_path}: {e}")

    print("\nDone.")
    print(f"  Found   : {len(files)}")
    print(f"  Wrote   : {wrote}")
    print(f"  Skipped : {skipped}  (already exists, use --overwrite)")
    print(f"  Errors  : {errors}")


if __name__ == "__main__":
    main()