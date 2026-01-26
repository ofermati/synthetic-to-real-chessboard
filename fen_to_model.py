import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fen", required=True)

    # Blender
    p.add_argument("--blender", default="./blender-3.6.5-linux-x64/blender")
    p.add_argument("--blend", required=True)

    # Model
    p.add_argument("--weights_dir", required=True)
    p.add_argument("--output_png", required=True)

    # Temp render folder
    p.add_argument("--render_dir", default="./temp_runs")

    args = p.parse_args()

    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    # ====================
    # 1. Run Blender
    # ====================
    blender_cmd = [
        args.blender,
        args.blend,
        "--background",
        "--python", "chess_position_api_v2_cropped_batch.py",
        "--",
        "--fen", args.fen,
        "--output_dir", str(render_dir),
    ]

    print("Running Blender...")
    subprocess.run(blender_cmd, check=True)

    # Expect this file from Blender:
    input_img = render_dir / "2_west.png"
    if not input_img.exists():
        raise RuntimeError(f"Expected render not found: {input_img}")

    # ====================
    # 2. Run model
    # ====================
    model_cmd = [
        "python",
        "run_model_on_image.py",
        "--input", str(input_img),
        "--weights_dir", args.weights_dir,
        "--output", args.output_png,
    ]

    print("Running model...")
    subprocess.run(model_cmd, check=True)

    print("✅ Done")
    print("Final image:", args.output_png)


if __name__ == "__main__":
    main()