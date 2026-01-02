import os
import csv
import chess.pgn

# =========================
# Path setup (DL_project)
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# DL_project = parent folder of "scripts"
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

PGN_DIR = os.path.join(PROJECT_DIR, "PGNs")
OUT_DIR = os.path.join(PROJECT_DIR, "fens")
OUT_CSV = os.path.join(OUT_DIR, "fens.csv")

EVERY_N_PLIES = 10  # 5 full moves = 10 plies

# =========================
# PGN processing
# =========================

def extract_from_pgn_file(pgn_path: str):
    """
    Returns list of rows:
    (pgn_file, game_index, ply_number, move_number, fen)
    """
    rows = []
    pgn_name = os.path.basename(pgn_path)

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        game_idx = 0

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_idx += 1
            board = game.board()
            ply = 0

            for move in game.mainline_moves():
                board.push(move)
                ply += 1

                if ply % EVERY_N_PLIES == 0:
                    move_number = (ply + 1) // 2
                    rows.append([
                        pgn_name,
                        game_idx,
                        ply,
                        move_number,
                        board.fen()
                    ])

    return rows

# =========================
# Main
# =========================

def main():
    print("📂 PGN_DIR:", PGN_DIR)
    print("📂 OUT_DIR:", OUT_DIR)

    if not os.path.isdir(PGN_DIR):
        raise FileNotFoundError(f"PGN folder not found: {PGN_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    pgn_files = sorted(
        fn for fn in os.listdir(PGN_DIR)
        if fn.lower().endswith(".pgn")
    )

    if not pgn_files:
        print("⚠️ No PGN files found.")
        return

    all_rows = []
    for fn in pgn_files:
        path = os.path.join(PGN_DIR, fn)
        all_rows.extend(extract_from_pgn_file(path))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow([
            "pgn_file",
            "game_index",
            "ply",
            "move_number",
            "fen"
        ])
        writer.writerows(all_rows)

    print(f"✅ Wrote {len(all_rows)} FEN rows to:")
    print(OUT_CSV)

if __name__ == "__main__":
    main()
