import bpy
import math
import os
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
import sys
import argparse

# ==========================
# CONFIG DEFAULTS
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 0.8
DESIRED_ANGLE_DEGREES = 30
LENS = 50
RES = 2048
SAMPLES = 128
OUT_DIR = "//renders"

# ==========================
# BOARD INFO
# ==========================
def get_board_info():
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    if plane is None or frame is None:
        raise RuntimeError('Missing objects: "Black & white" and/or "Outer frame"')

    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8.0

    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2.0
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)
    scale_factor = board_size / REAL_BOARD_SIZE

    return {
        "square_size": square_size,
        "plane_min": plane_min,
        "plane_max": plane_max,
        "center": center,
        "scale_factor": scale_factor,
    }

def position_to_square(pos, board_info):
    square_size = board_info["square_size"]
    plane_min = board_info["plane_min"]
    plane_max = board_info["plane_max"]

    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord("a") + file_idx)

    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1
    return f"{file_letter}{rank_number}"

# ==========================
# PIECES + FEN
# ==========================
def detect_starting_positions(board_info):
    pieces = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue

        name = obj.name
        piece_type = None

        if name in ["B", "C", "D", "E", "F", "G", "H", "A(texture)"]:
            piece_type = "P"
        elif name in ["B.001", "C.001", "D.001", "E.001", "F.001", "G.001", "H.001", "A(textures)"]:
            piece_type = "p"
        elif "rook" in name.lower():
            piece_type = "R" if "white" in name.lower() else "r"
        elif "knight" in name.lower():
            piece_type = "N" if "white" in name.lower() else "n"
        elif "bishop" in name.lower() or "bitshop" in name.lower():
            piece_type = "B" if "white" in name.lower() else "b"
        elif "queen" in name.lower():
            piece_type = "Q" if "white" in name.lower() else "q"
        elif "king" in name.lower():
            piece_type = "K" if "white" in name.lower() else "k"

        if piece_type:
            square = position_to_square(obj.location, board_info)
            pieces[name] = {"square": square, "piece_type": piece_type}

    return pieces

def parse_fen(fen):
    board_fen = fen.split()[0]
    ranks = board_fen.split("/")
    position = {}

    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx
        for ch in rank:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                file_letter = chr(ord("a") + file_idx)
                position[f"{file_letter}{board_rank}"] = ch
                file_idx += 1

    return position

def apply_fen(fen, starting_pieces, board_info):
    target = parse_fen(fen)
    square_size = board_info["square_size"]
    used = set()

    for target_sq, piece_type in target.items():
        candidates = []
        for name, info in starting_pieces.items():
            if info["piece_type"] != piece_type or name in used:
                continue

            from_sq = info["square"]
            from_file = ord(from_sq[0]) - ord("a")
            from_rank = int(from_sq[1]) - 1
            to_file = ord(target_sq[0]) - ord("a")
            to_rank = int(target_sq[1]) - 1
            dist = abs(to_file - from_file) + abs(to_rank - from_rank)
            candidates.append((dist, name, from_sq))

        if not candidates:
            continue

        candidates.sort()
        _, piece_name, from_sq = candidates[0]
        obj = bpy.data.objects.get(piece_name)
        if not obj:
            continue

        from_file = ord(from_sq[0]) - ord("a")
        from_rank = int(from_sq[1]) - 1
        to_file = ord(target_sq[0]) - ord("a")
        to_rank = int(target_sq[1]) - 1

        obj.location.x -= (to_file - from_file) * square_size
        obj.location.y -= (to_rank - from_rank) * square_size
        obj.hide_render = False
        obj.hide_viewport = False
        used.add(piece_name)

    for piece_name in starting_pieces.keys():
        if piece_name not in used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True

# ==========================
# BORDER CROP
# ==========================
def ndc_bounds(scene, cam, obj):
    xs, ys = [], []
    for v in obj.bound_box:
        w = obj.matrix_world @ Vector(v)
        co = world_to_camera_view(scene, cam, w)
        xs.append(co.x)
        ys.append(co.y)
    return min(xs), max(xs), min(ys), max(ys)

def set_border(scene, min_x, max_x, min_y, max_y, margin):
    scene.render.use_border = True
    scene.render.use_crop_to_border = True
    scene.render.border_min_x = max(0.0, min_x - margin)
    scene.render.border_max_x = min(1.0, max_x + margin)
    scene.render.border_min_y = max(0.0, min_y - margin)
    scene.render.border_max_y = min(1.0, max_y + margin)

def clear_border(scene):
    scene.render.use_border = False
    scene.render.use_crop_to_border = False

# ==========================
# RENDER
# ==========================
def render_all_views(board_info, view="black", crop_margin=0.06):
    """
    Render 3 views and crop so the board/frame fills the whole image.

    Outputs (teacher-style names so your pipeline keeps working):
      - 1_overhead.png
      - (black) 2_west.png, 3_east.png
      - (white) 2_east.png, 3_west.png
    """
    print("\n" + "=" * 70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("=" * 70)

    center = board_info["center"]
    scale_factor = board_info["scale_factor"]

    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)
    camera_z = center.z + camera_height

    # Remove existing cameras (avoid accumulation)
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Ensure light exists
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light_height = center.z + camera_height * 2
        bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
        bpy.context.active_object.data.energy = 3.0

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
    except Exception:
        pass

    # Pick object to crop around (frame is best)
    frame = bpy.data.objects.get("Outer frame")
    if frame is None:
        raise RuntimeError('Missing object: "Outer frame" (needed for cropping)')

    # View definitions
    if view == "white":
        views = [
            ((center.x, center.y, camera_z), "1_overhead.png"),
            ((center.x + horizontal_offset, center.y, camera_z), "2_east.png"),
            ((center.x - horizontal_offset, center.y, camera_z), "3_west.png"),
        ]
        z_rotation_offset = math.radians(180)
    else:  # black
        views = [
            ((center.x, center.y, camera_z), "1_overhead.png"),
            ((center.x - horizontal_offset, center.y, camera_z), "2_west.png"),
            ((center.x + horizontal_offset, center.y, camera_z), "3_east.png"),
        ]
        z_rotation_offset = 0.0

    for location, filename in views:
        print(f"\nRendering: {filename}")

        # Create camera
        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object

        # Always look at the board center (THIS FIXES YOUR "STRIP" BUG)
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        # Apply white/black rotation
        cam.rotation_euler.z += z_rotation_offset

        cam.data.lens = LENS
        scene.camera = cam

        # Crop-to-frame using normalized camera coords
        min_x, max_x, min_y, max_y = ndc_bounds(scene, cam, frame)
        set_border(scene, min_x, max_x, min_y, max_y, margin=crop_margin)

        # Render
        scene.render.filepath = f"{OUT_DIR}/{filename}"
        bpy.ops.render.render(write_still=True)

        # Clear border so it won't stick to the next render
        clear_border(scene)

        print(f"  ✓ Saved: {filename}")

        # Remove camera
        bpy.data.objects.remove(cam, do_unlink=True)

    print("\n✓ Rendering complete")


# ==========================
# MAIN
# ==========================
def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, required=True)
    parser.add_argument("--view", type=str, default="black", choices=["white", "black"])
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="//renders")
    parser.add_argument("--crop_margin", type=float, default=0.06)
    args = parser.parse_args(argv)

    global RES, SAMPLES, OUT_DIR
    RES = args.resolution
    SAMPLES = args.samples
    OUT_DIR = bpy.path.abspath(args.output_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    board_info = get_board_info()
    starting = detect_starting_positions(board_info)
    apply_fen(args.fen, starting, board_info)
    render_all_views(board_info, view=args.view, crop_margin=args.crop_margin)

if __name__ == "__main__":
    main()
