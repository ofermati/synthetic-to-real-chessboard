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

FENS_DIR    = Path(r"C:\Dl_project\fens")
OUT_BASE    = Path(r"C:\Dl_project\renders_one")   # שמרתי בתיקייה חדשה כדי לא לדרוס
GLOBAL_RENDERS_DIR = Path(r"C:\renders")

# =========================
# RENDER PARAMS
# =========================
RESOLUTION  = 1024
SAMPLES     = 64
LIMIT       = None  # None = הכל

# =========================
# בחירת זווית לכל משחק
# view: "white" או "black"
# angle: "overhead" או "2" או "3"
# =========================
GAME_VIEW_ANGLE = {
    "Game2": ("white", "2"),
    "Game4": ("white", "3"),
    "Game5": ("white", "2"),
    "Game6": ("white", "1"),  # overhead
    "Game7": ("white", "3"),
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

def collect_one_from_global(sample_dir: Path, view: str, angle: str) -> None:
    """
    שומר רק תמונה אחת:
    angle:
      "overhead" -> 1_overhead.png
      "2"        -> (2_west או 2_east) לפי מה שקיים
      "3"        -> (3_east או 3_west) לפי מה שקיים
    נשמר בשם: chosen.png
    """
    if angle == "overhead":
        src = GLOBAL_RENDERS_DIR / "1_overhead.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing 1_overhead.png in {GLOBAL_RENDERS_DIR}")

    elif angle == "2":
        src = pick_existing(
            [GLOBAL_RENDERS_DIR / "2_west.png", GLOBAL_RENDERS_DIR / "2_east.png"],
            "2_west.png or 2_east.png"
        )

    elif angle == "3":
        src = pick_existing(
            [GLOBAL_RENDERS_DIR / "3_east.png", GLOBAL_RENDERS_DIR / "3_west.png"],
            "3_east.png or 3_west.png"
        )
    else:
        raise ValueError(f"Invalid angle '{angle}'. Use: overhead / 2 / 3")

    dst = sample_dir / "chosen.png"
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))

def is_done(sample_dir: Path) -> bool:
    return (sample_dir / "chosen.png").exists()

def make_sample_id_from_to_frame(row: dict) -> str:
    to_frame = row.get("to_frame")
    if to_frame is None or str(to_frame).strip() == "":
        raise ValueError("Missing to_frame in CSV row")
    frame_num = int(str(to_frame).strip())
    return f"frame_{frame_num:06d}"

def read_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
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

    csv_files = sorted(FENS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {FENS_DIR}")

    print(f"Found {len(csv_files)} CSV files in {FENS_DIR}")

    for csv_path in csv_files:
        game_name = csv_path.stem  # Game2 וכו'
        view, angle = GAME_VIEW_ANGLE.get(game_name, DEFAULT_VIEW_ANGLE)

        out_root = OUT_BASE / game_name
        out_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"PROCESSING: {csv_path.name}")
        print(f"CHOICE: view={view}, angle={angle}")
        print(f"OUTPUT DIR: {out_root}")
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

            sample_dir = out_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            if is_done(sample_dir):
                skipped += 1
                print(f"[{i}/{len(rows)}] SKIP (already done): {game_name}/{sample_id}")
                continue

            print(f"[{i}/{len(rows)}] RENDER: {game_name}/{sample_id}")

            try:
                clear_global_renders()
                run_blender_one_view(fen, view, sample_dir)
                collect_one_from_global(sample_dir, view, angle)
                ok += 1
            except Exception as e:
                failed += 1
                print("❌ FAILED:", f"{game_name}/{sample_id}")
                print(str(e)[:2000])

        print("\n---- GAME SUMMARY ----")
        print("OK:", ok)
        print("SKIPPED:", skipped)
        print("FAILED:", failed)

if __name__ == "__main__":
    main()
