import csv
import os
import subprocess
from pathlib import Path
import shutil

# =========================
# PATHS
# =========================

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE  = r"C:\Dl_project\blender\chess-set.blend"
BLENDER_PY  = r"C:\Dl_project\blender\chess_position_api_v2.py"

# Folder that contains ALL Game*.csv files
FENS_DIR    = Path(r"C:\Dl_project\fens")

# Base output folder; each CSV will create its own subfolder inside (Game4, Game5, ...)
OUT_BASE    = Path(r"C:\Dl_project\renders")

# Blender actually writes here on your machine
GLOBAL_RENDERS_DIR = Path(r"C:\renders")

# =========================
# RENDER PARAMS
# =========================

RESOLUTION  = 1024
SAMPLES     = 64

# Set to None to render ALL rows in each CSV
LIMIT       = None  # example: 4 renders only first 4 rows per CSV

VIEWS = ["black", "white"]

# =========================

def clear_global_renders() -> None:
    """Delete pngs from C:\renders so we don't mix runs."""
    GLOBAL_RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    for p in GLOBAL_RENDERS_DIR.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass

def run_blender_one_view(fen: str, view: str, workdir: Path) -> None:
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

def collect_from_global(sample_dir: Path, view: str) -> None:
    """
    Collect Blender outputs from C:\renders and move into sample_dir with stable names.
    Lecturer output files:
      always: 1_overhead.png
      black:  2_west.png,  3_east.png
      white:  2_east.png,  3_west.png

    We normalize to:
      {view}_1_overhead.png
      {view}_2.png
      {view}_3.png
    """
    overhead = GLOBAL_RENDERS_DIR / "1_overhead.png"
    p2 = pick_existing(
        [GLOBAL_RENDERS_DIR / "2_west.png", GLOBAL_RENDERS_DIR / "2_east.png"],
        "2_west.png or 2_east.png"
    )
    p3 = pick_existing(
        [GLOBAL_RENDERS_DIR / "3_east.png", GLOBAL_RENDERS_DIR / "3_west.png"],
        "3_east.png or 3_west.png"
    )

    if not overhead.exists():
        raise FileNotFoundError(f"Missing 1_overhead.png in {GLOBAL_RENDERS_DIR}")

    mapping = {
        overhead: sample_dir / f"{view}_1_overhead.png",
        p2:       sample_dir / f"{view}_2.png",
        p3:       sample_dir / f"{view}_3.png",
    }

    for src, dst in mapping.items():
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))

def is_done(sample_dir: Path) -> bool:
    for view in VIEWS:
        for name in ["1_overhead.png", "2.png", "3.png"]:
            if not (sample_dir / f"{view}_{name}").exists():
                return False
    return True

def make_sample_id_from_to_frame(row: dict) -> str:
    """
    Build stable sample id from CSV's to_frame.
    Example: to_frame=588 -> frame_000588
    """
    to_frame = row.get("to_frame")
    if to_frame is None or str(to_frame).strip() == "":
        raise ValueError("Missing to_frame in CSV row")
    frame_num = int(str(to_frame).strip())
    return f"frame_{frame_num:06d}"

def read_csv_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

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

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Find all CSVs in fens folder
    csv_files = sorted(FENS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {FENS_DIR}")

    print(f"Found {len(csv_files)} CSV files in {FENS_DIR}")

    grand_ok = grand_skipped = grand_failed = 0

    for csv_path in csv_files:
        game_name = csv_path.stem  # e.g., "Game4"
        out_root = OUT_BASE / game_name
        out_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"PROCESSING: {csv_path.name}")
        print(f"OUTPUT DIR: {out_root}")
        print("=" * 60)

        rows = read_csv_rows(csv_path)
        if LIMIT is not None:
            rows = rows[:LIMIT]

        print(f"Rows: {len(rows)}")

        ok = skipped = failed = 0

        for i, r in enumerate(rows, start=1):
            if "fen" not in r or not r["fen"]:
                print(f"[{i}/{len(rows)}] SKIP – missing fen")
                skipped += 1
                continue

            fen = r["fen"].strip()

            try:
                sample_id = make_sample_id_from_to_frame(r)
            except Exception as e:
                failed += 1
                print(f"❌ FAILED row {i}: can't build sample_id: {e}")
                continue

            sample_dir = out_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            if is_done(sample_dir):
                skipped += 1
                print(f"[{i}/{len(rows)}] SKIP (already done): {game_name}/{sample_id}")
                continue

            print(f"[{i}/{len(rows)}] RENDER: {game_name}/{sample_id}")

            try:
                for view in VIEWS:
                    clear_global_renders()
                    run_blender_one_view(fen, view, sample_dir)
                    collect_from_global(sample_dir, view)

                ok += 1

            except Exception as e:
                failed += 1
                print("❌ FAILED:", f"{game_name}/{sample_id}")
                print(str(e)[:2000])

        print("\n---- GAME SUMMARY ----")
        print("OK:", ok)
        print("SKIPPED:", skipped)
        print("FAILED:", failed)

        grand_ok += ok
        grand_skipped += skipped
        grand_failed += failed

    print("\n" + "#" * 60)
    print("==== GRAND SUMMARY (ALL GAMES) ====")
    print("OK:", grand_ok)
    print("SKIPPED:", grand_skipped)
    print("FAILED:", grand_failed)
    print("Output base:", str(OUT_BASE))
    print("#" * 60)

if __name__ == "__main__":
    main()
