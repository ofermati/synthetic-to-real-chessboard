from pathlib import Path
from PIL import Image
import re

FRAME_DIR = Path("outputs/cyclegan_run1/infer_s2r_east")
OUT_PATH  = Path("board_full.png")

rc_re = re.compile(r"_r(\d+)_c(\d+)\.png$", re.IGNORECASE)

files = list(FRAME_DIR.glob("*.png"))
if not files:
    raise SystemExit(f"No png files in {FRAME_DIR}")

# index: (r,c) -> filepath
tile_map = {}
for f in files:
    m = rc_re.search(f.name)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        tile_map[(r, c)] = f

if not tile_map:
    raise SystemExit("No files matched pattern *_r#_c#.png")

# גודל tile מהראשון
first_tile = Image.open(next(iter(tile_map.values()))).convert("RGB")
tile_w, tile_h = first_tile.size

max_r = max(r for r, _ in tile_map.keys())
max_c = max(c for _, c in tile_map.keys())
rows, cols = max_r + 1, max_c + 1

canvas = Image.new("RGB", (cols * tile_w, rows * tile_h))

missing = []
for r in range(rows):
    for c in range(cols):
        f = tile_map.get((r, c))
        if f is None:
            missing.append((r, c))
            continue
        img = Image.open(f).convert("RGB")
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.BILINEAR)
        canvas.paste(img, (c * tile_w, r * tile_h))

canvas.save(OUT_PATH)
print("Saved:", OUT_PATH)
if missing:
    print("Missing tiles:", missing)
