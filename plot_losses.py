import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(csv_path: str, out_dir: str = "plots", dedup_by_epoch: bool = True):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "epoch" not in df.columns:
        raise ValueError("CSV must contain an 'epoch' column")

    # Sort (and dedup if needed)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["epoch", "timestamp"])
    else:
        df = df.sort_values(["epoch"])

    if dedup_by_epoch:
        df = df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)

    # Convert numeric columns
    for c in df.columns:
        if c not in ("timestamp",):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    x = df["epoch"]
    os.makedirs(out_dir, exist_ok=True)

    # Helper: plot only existing cols
    def plot_cols(title, fname, cols_and_labels):
        plt.figure()
        plotted = False
        for col, label in cols_and_labels:
            if col in df.columns and df[col].notna().any():
                plt.plot(x, df[col], label=label)
                plotted = True
        if not plotted:
            plt.close()
            print(f"[WARN] Skipping {fname}: none of the requested columns exist in the CSV.")
            return
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    # === For your CSV: epoch, loss_G, loss_D, loss_NCE, loss_ID ===
    plot_cols(
        title="Training losses (G / D)",
        fname="loss_G_D.png",
        cols_and_labels=[
            ("loss_G", "loss_G"),
            ("loss_D", "loss_D"),
        ],
    )

    plot_cols(
        title="Additional losses (NCE / ID)",
        fname="loss_NCE_ID.png",
        cols_and_labels=[
            ("loss_NCE", "loss_NCE"),
            ("loss_ID", "loss_ID"),
        ],
    )

    # Optional: if later you add CycleGAN-style columns, this will automatically work
    plot_cols(
        title="Cycle / Identity losses (if available)",
        fname="cycle_identity.png",
        cols_and_labels=[
            ("loss_cycle", "cycle"),
            ("loss_id", "identity"),
        ],
    )

    print(f"Saved plots to ./{out_dir}/")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_losses.py <csv>")
        raise SystemExit(2)

    plot_losses(sys.argv[1])