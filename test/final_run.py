import argparse
import subprocess
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fen", required=True)

    # --- 1. Dynamic path resolution (make script portable) ---
    # Locate current script
    current_script_path = Path(__file__).resolve()
    # Assume ofer.py is inside test/, so go two levels up to project root
    project_root = current_script_path.parent.parent 
    
    # Blender path in home directory
    default_blender = Path.home() / "blender-3.6.5-linux-x64/blender"

    # --- Arguments ---
    p.add_argument("--blender", default=str(default_blender))
    
    # Important: paths point into blender/ directory
    p.add_argument("--blend", default=str(project_root / "blender/chess-set.blend"))
    p.add_argument("--blender_script", default=str(project_root / "blender/chess_position_api_v2_cropped_batch.py"))

    p.add_argument("--view", choices=["black", "white"], default="black")
    p.add_argument("--resolution", type=int, default=2048)
    p.add_argument("--samples", type=int, default=256)

    # Zoom-crop script
    p.add_argument("--zoom_script", default=str(project_root / "data_scripts/zoom_remove_frame.py"))
    p.add_argument("--zoom", type=float, default=0.9, help="Lower number = more zoom (cut more border)")
    p.add_argument("--overwrite_zoom", action="store_true")

    # Output paths
    p.add_argument("--render_dir", default="./temp_runs/renders_raw")
    p.add_argument("--zoomed_dir", default="./temp_runs/renders_zoomed")

    # Model
    p.add_argument("--model_script", default=str(project_root / "test/new_8X8_to_full.py"))
    p.add_argument("--weights_dir", default=str(project_root / "outputs/cut_1_8X8_new/weights"))
    p.add_argument("--output_png", default="data_test/result.png")

    # Which render to use (from blender output)
    p.add_argument("--render_pick", default="2_west.png")

    args = p.parse_args()
    
    # Create output directories
    render_dir = Path(args.render_dir)
    zoomed_dir = Path(args.zoomed_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    zoomed_dir.mkdir(parents=True, exist_ok=True)

    # ====================
    # 1) Run Blender: FEN -> images in render_dir
    # ====================
    blender_cmd = [
        str(Path(args.blender)),
        str(Path(args.blend)),
        "--background",
        "--python", str(Path(args.blender_script)),
        "--",
        "--fen", args.fen,
        "--view", args.view,
        "--resolution", str(args.resolution),
        "--samples", str(args.samples),
        "--output_dir", str(render_dir),
    ]
    
    print("\n--- Step 1: Running Blender ---")
    print(f"DEBUG: Using blend file: {args.blend}")
    subprocess.run(blender_cmd, check=True)

    # Verify raw image was created
    raw_img = render_dir / args.render_pick
    if not raw_img.exists():
        raise RuntimeError(
            f"Expected render not found: {raw_img}\n"
            f"Check --render_dir/--render_pick and what Blender actually outputs."
        )

    # ====================
    # 2) Run zoom script: render_dir -> zoomed_dir
    # ====================
    zoom_cmd = [
        sys.executable,  # Use current Python interpreter (important!)
        str(Path(args.zoom_script)),
        "--in-root", str(render_dir),
        "--out-root", str(zoomed_dir),
        "--zoom", str(args.zoom),
        "--overwrite"  # Force overwrite to ensure it always runs
    ]

    print("\n--- Step 2: Running Zoom Script ---")
    print(f"DEBUG: Zoom command: {' '.join(zoom_cmd)}")
    
    try:
        # capture_output=True helps catch errors on failure
        subprocess.run(zoom_cmd, check=True, capture_output=True, text=True)
        print("✅ Zoom script finished successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error running zoom script:")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise e

    # Verify zoomed image was created
    zoomed_img = zoomed_dir / args.render_pick
    if not zoomed_img.exists():
        raise RuntimeError(
            f"Expected zoomed image not found: {zoomed_img}\n"
            f"Zoom step ran but didn't produce the expected file."
        )

    # ====================
    # 3) Run model on zoomed image
    # ====================
    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    model_cmd = [
        sys.executable,  # Use current Python here as well
        str(Path(args.model_script)),
        "--input", str(zoomed_img),
        "--weights_dir", str(Path(args.weights_dir)),
        "--output", str(output_png),
    ]
    
    print("\n--- Step 3: Running Model ---")
    subprocess.run(model_cmd, check=True)

    print("\n✅ ALL DONE")
    print(f"1. Raw render   : {raw_img}")
    print(f"2. Zoomed image : {zoomed_img}")
    print(f"3. Final output : {output_png}")


if __name__ == "__main__":
    main()