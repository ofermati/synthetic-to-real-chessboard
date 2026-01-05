import csv
import os
import shutil
import subprocess
from pathlib import Path

# =========================
# PATHS (Cluster-friendly)
# =========================

def resolve_blender_exe(project_root: Path) -> str:
    """
    Finds Blender executable.
    Priority:
      1) BLENDER_EXE env var
      2) blender in PATH
      3) common local candidates
    """
    env = os.environ.get("BLENDER_EXE")
    if env and Path(env).exists():
        return env

    p = shutil.which("blender")
    if p:
        return p

    candidates = [
        Path.home() / "blender-5.0.0-linux-x64" / "blender",
        Path.home() / "blender-3.6.5-linux-x64" / "blender",
        Path.home() / "apps" / "blender-3.6.5-linux-x64" / "blender",
        project_root / "blender_bin" / "blender",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "Blender executable not found.\n"
        "Fix options:\n"
        "  (A) export BLENDER_EXE=/full/path/to/blender\n"
        "  (B) add blender folder to PATH\n"
        "Tried: " + ", ".join(str(x) for x in candidates)
    )


# project root = synthetic-to-real-chessboard
PROJECT_ROOT = Path("/home/nitzandu/synthetic-to-real-chessboard")

BLENDER_EXE = resolve_blender_exe(PROJECT_ROOT)
BLEND_FILE  = str(PROJECT_ROOT / "blender" / "chess-set.blend")
BLENDER_PY  = str(PROJECT_ROOT / "blender" / "chess_position_api_v2.py")

# Input CSVs (absolute, as requested)
FENS_DIR = Path("/home/nitzandu/synthetic-to-real-chessboard/fens")

# Output paired dataset (we create only synthetic; you copy real manually)
PAIRED_DIR = PROJECT_ROOT / "datasets" / "paired"
OUT_SYN_BASE  = PAIRED_DIR / "synthetic"

# Temporary Blender outputs (avoid mixing runs)
# Temporary data root
TEMP_DATA_DIR = PROJECT_ROOT / "temp_data"

# Final output directory (NO synthetic subfolder)
RENDERS_PAIRS_DIR = TEMP_DATA_DIR / "renders_pairs"

# Temporary Blender outputs
GLOBAL_RENDERS_DIR = TEMP_DATA_DIR / "temp_renders"

# =========================
# RENDER PARAMS
# =========================
RESOLUTION  = 1024
SAMPLES     = 64
LIMIT       = None  # None = render all rows

# =========================
# angle selection per game
# view: "white" or "black"
# angle: "overhead" / "2" / "3"
# =========================
GAME_VIEW_ANGLE = {
    "game2": ("white", "2"),
    "game4": ("white", "3"),
    "game5": ("white", "2"),
    "game6": ("white", "overhead"),
    "game7": ("white", "3"),
}

DEFAULT_VIEW_ANGLE = ("white", "2")

# =========================

def clear_global_renders() -> None:
    GLOBAL_RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    for p in GLOBAL_RENDERS_DIR.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass

def run_blender_one_view(fen: str, view: str, workdir: Path) -> None:
    """
    Calls your Blender python script once (for one FEN + one view).
    That script usually writes multiple pngs (overhead + east/west).
    We'll pick exactly ONE of them.
    """
    cmd = [
        BLENDER_EXE,
        BLEND_FILE,
        "--background",
        "--python-exit-code", "1",
        "--python",
        BLENDER_PY,
        "--",
        "--fen", fen,
        "--view", view,
        "--resolution", str(RESOLUTION),
        "--samples", str(SAMPLES),
        "--output_dir", str(GLOBAL_RENDERS_DIR),
    ]

    res = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if res.returncode != 0:
        raise RuntimeError(
            f"Blender failed (view={view}).\n"
            f"STDOUT (tail):\n{(res.stdout or '')[-2000:]}\n"
            f"STDERR (tail):\n{(res.stderr or '')[-2000:]}"
        )

def pick_existing(paths, label):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {label}. Tried: {[str(x) for x in paths]}")

def read_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def make_sample_id_from_to_frame(row: dict) -> str:
    """
    You said real frames are named like: frame_000200.jpg etc.
    We'll use to_frame to generate same naming.
    """
    to_frame = row.get("to_frame")
    if to_frame is None or str(to_frame).strip() == "":
        raise ValueError("Missing to_frame in CSV row")
    frame_num = int(str(to_frame).strip())
    return f"frame_{frame_num:06d}"

def choose_one_render_png(angle: str) -> Path:
    """
    Blender script output files commonly:
      - 1_overhead.png always
      - 2_west.png / 2_east.png
      - 3_west.png / 3_east.png

    We choose exactly ONE according to angle:
      "overhead" -> 1_overhead.png
      "2"        -> whichever exists: 2_west/2_east
      "3"        -> whichever exists: 3_west/3_east
    """
    if angle == "overhead":
        src = GLOBAL_RENDERS_DIR / "1_overhead.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing 1_overhead.png in {GLOBAL_RENDERS_DIR}")
        return src

    if angle == "2":
        return pick_existing(
            [GLOBAL_RENDERS_DIR / "2_west.png", GLOBAL_RENDERS_DIR / "2_east.png"],
            "2_west.png or 2_east.png"
        )

    if angle == "3":
        return pick_existing(
            [GLOBAL_RENDERS_DIR / "3_west.png", GLOBAL_RENDERS_DIR / "3_east.png"],
            "3_west.png or 3_east.png"
        )

    raise ValueError(f"Invalid angle '{angle}'. Use: overhead / 2 / 3")

def save_png_as_jpg(png_path: Path, jpg_path: Path) -> None:
    """
    Save output as JPG: frame_XXXXXX.jpg
    Preferred: Pillow conversion.
    If Pillow missing, fallback copies bytes (not ideal). We'll warn.
    """
    jpg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image  # pillow
        img = Image.open(png_path).convert("RGB")
        img.save(jpg_path, quality=95)
    except Exception as e:
        print(f"⚠️ Pillow convert failed ({e}). Copying PNG bytes into .jpg (not ideal).")
        shutil.copy2(png_path, jpg_path)

def main():
    # sanity checks
    for p, label in [
        (BLENDER_EXE, "BLENDER_EXE"),
        (BLEND_FILE,  "BLEND_FILE"),
        (BLENDER_PY,  "BLENDER_PY"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    if not FENS_DIR.exists():
        raise FileNotFoundError(f"FENS_DIR not found: {FENS_DIR}")

    RENDERS_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(FENS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {FENS_DIR}")

    print(f"Found {len(csv_files)} CSV files in {FENS_DIR}")
    print(f"Output synthetic paired dataset: {OUT_SYN_BASE}")
    print("NOTE: real images are NOT copied by this script (you will copy them manually).")

    for csv_path in csv_files:
        game_name = csv_path.stem  # e.g. Game2
        view, angle = GAME_VIEW_ANGLE.get(game_name, DEFAULT_VIEW_ANGLE)

        out_syn_game = RENDERS_PAIRS_DIR / game_name
        out_syn_game.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"PROCESSING: {csv_path.name}")
        print(f"CHOICE: view={view}, angle={angle}")
        print(f"OUT SYN : {out_syn_game}")
        print("=" * 60)

        rows = read_csv_rows(csv_path)
        if LIMIT is not None:
            rows = rows[:LIMIT]

        ok = skipped = failed = 0

        for i, r in enumerate(rows, start=1):
            fen = (r.get("fen") or "").strip()
            if not fen:
                skipped += 1
                print(f"[{i}/{len(rows)}] SKIP – missing fen")
                continue

            try:
                sample_id = make_sample_id_from_to_frame(r)
            except Exception as e:
                failed += 1
                print(f"❌ FAILED row {i}: {e}")
                continue

            dst_syn = out_syn_game / f"{sample_id}.jpg"

            if dst_syn.exists():
                skipped += 1
                print(f"[{i}/{len(rows)}] SKIP (exists): {game_name}/{sample_id}")
                continue

            try:
                clear_global_renders()
                run_blender_one_view(fen, view, workdir=PROJECT_ROOT)

                src_png = choose_one_render_png(angle)
                save_png_as_jpg(src_png, dst_syn)

                ok += 1
                print(f"[{i}/{len(rows)}] OK: {game_name}/{sample_id}")
            except Exception as e:
                failed += 1
                print(f"❌ FAILED: {game_name}/{sample_id}")
                print(str(e)[:2000])

        print("\n---- GAME SUMMARY ----")
        print("OK:", ok)
        print("SKIPPED:", skipped)
        print("FAILED:", failed)

if __name__ == "__main__":
    main()
