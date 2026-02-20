#!/usr/bin/env python
"""
Demo: Segmentation + Inverse Perspective Mapping (IPM) bird's-eye view in CARLA.
Supports both SegFormer (multi-class) and DeepLabV3+ ORFD (binary freespace).

Shows three panels side-by-side:
    [Camera feed] [BEV segmentation] [Legend]

The BEV panel warps the segmentation mask onto a top-down ground plane using a
homography computed from the camera intrinsics and mount position. A grid overlay
shows distances in meters.

Car drives on autopilot. Requires: CARLA running, pip install -r requirements-segmentation.txt.

Usage:
    python scripts/autopilot_ground_projection.py
    python scripts/autopilot_ground_projection.py --model training/models/rellis3d_b0_ade_arc
    python scripts/autopilot_ground_projection.py --deeplab
    python scripts/autopilot_ground_projection.py --bev-x 8 --bev-ymin 5 --bev-ymax 30
    Press 'q' or ESC to exit.
"""

import argparse
import math
import random
import sys
import threading
import time

try:
    import carla
except ImportError:
    print("CARLA Python package not found. Install the wheel from your CARLA install.")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from torchvision import transforms
except ImportError as e:
    print("Missing dependency: %s" % e)
    print("Install: pip install -r requirements-segmentation.txt  and  pip install opencv-python")
    sys.exit(1)

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    _HAS_SMP = False


# Hand-picked visually distinct colors for off-road classes (BGR order for OpenCV).
OFFROAD_COLORS_BGR = {
    "dirt":       (43, 90, 139),
    "grass":      (0, 160, 0),
    "tree":       (34, 120, 34),
    "pole":       (50, 50, 255),
    "water":      (220, 100, 30),
    "sky":        (235, 206, 135),
    "vehicle":    (200, 0, 200),
    "object":     (70, 130, 180),
    "asphalt":    (80, 80, 80),
    "building":   (150, 150, 150),
    "log":        (0, 50, 100),
    "person":     (100, 0, 255),
    "fence":      (220, 150, 190),
    "bush":       (100, 180, 0),
    "concrete":   (170, 170, 170),
    "barrier":    (0, 200, 255),
    "puddle":     (180, 50, 50),
    "mud":        (30, 60, 90),
    "rubble":     (90, 120, 160),
    "sign":       (50, 220, 255),
    "rock":       (100, 128, 128),
    "sand":       (160, 200, 220),
    "gravel":     (130, 170, 180),
    "mulch":      (40, 100, 160),
    "rock-bed":   (80, 100, 110),
    "bridge":     (140, 140, 100),
    "trash":      (50, 200, 200),
    "bicycle":    (200, 100, 255),
}

LEGEND_CLASSES = frozenset([
    "road", "sidewalk", "building", "sky", "tree", "grass", "earth", "path",
    "car", "vehicle", "person", "pole", "fence", "wall", "plant", "bus", "truck",
    "bicycle", "motorcycle", "traffic light", "traffic sign", "bridge", "water",
    "rock", "stone", "sand", "ground", "terrain", "vegetation", "house",
    "mountain", "sea", "field", "runway", "river", "tower", "skyscraper",
    "floor", "pavement", "dirt", "mud", "snow", "lane", "dirt track",
])


# ---------------------------------------------------------------------------
# Palette helpers (same as autopilot_segformer.py)
# ---------------------------------------------------------------------------

def _generic_palette():
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        palette[i] = [(37 * i) % 256, (97 * i + 31) % 256, (157 * i + 67) % 256]
    return palette


def _offroad_palette_bgr(id2label, num_entries=256):
    palette = np.zeros((num_entries, 3), dtype=np.uint8)
    used = set()
    for idx, name in id2label.items():
        key = name.lower().strip()
        if key in OFFROAD_COLORS_BGR:
            palette[idx] = OFFROAD_COLORS_BGR[key]
            used.add(idx)
    for idx in range(num_entries):
        if idx not in used:
            hue = int((idx * 137.508) % 180)
            hsv = np.uint8([[[hue, 200, 220]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            palette[idx] = bgr
    return palette


def carla_image_to_bgr(carla_image):
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


# ---------------------------------------------------------------------------
# Legend (same as autopilot_segformer.py)
# ---------------------------------------------------------------------------

def build_legend(id2label, palette, height, is_finetuned, palette_is_bgr):
    legend_width = 220
    if is_finetuned:
        num_items = len(id2label) if id2label else 0
        if num_items == 0:
            legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 240
            cv2.putText(legend, "No labels", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return legend
        header_h = 26
        row_h = max(10, (height - header_h) // num_items)
        font_scale = 0.35 if row_h >= 14 else 0.28
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        cv2.putText(legend, "Class (id)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        items = sorted(id2label.items())
        for i, (idx, name) in enumerate(items):
            y = header_h + 10 + i * row_h
            if y + 4 > height:
                break
            color = palette[idx % 256]
            if palette_is_bgr:
                color_bgr = (int(color[0]), int(color[1]), int(color[2]))
            else:
                color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            box_h = min(row_h - 2, 14)
            cv2.rectangle(legend, (8, y - box_h), (8 + 16, y), color_bgr, -1)
            cv2.rectangle(legend, (8, y - box_h), (8 + 16, y), (80, 80, 80), 1)
            label = "%s (%d)" % (name[:14] if len(name) > 14 else name, idx)
            cv2.putText(legend, label, (30, y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    else:
        filtered = {i: n for i, n in id2label.items() if n.lower().strip() in LEGEND_CLASSES}
        display_labels = filtered if filtered else id2label
        num_rows = 20
        row_h = max(18, height // num_rows)
        actual_rows = min(num_rows, len(display_labels)) if display_labels else 0
        if actual_rows == 0:
            legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 240
            cv2.putText(legend, "No labels", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return legend
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        cv2.putText(legend, "Class (id)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        items = sorted(display_labels.items())[:actual_rows]
        for i, (idx, name) in enumerate(items):
            y = 36 + i * row_h
            if y + row_h > height:
                break
            color = palette[idx % 256]
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(legend, (8, y - 14), (8 + 20, y + 2), color_bgr, -1)
            cv2.rectangle(legend, (8, y - 14), (8 + 20, y + 2), (80, 80, 80), 1)
            label = (name[:18] + "..") if len(name) > 18 else name
            cv2.putText(legend, label, (34, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return legend


# ---------------------------------------------------------------------------
# IPM (Inverse Perspective Mapping)
# ---------------------------------------------------------------------------

def get_intrinsic_matrix(width, height, fov_deg):
    """Compute camera intrinsic matrix from CARLA camera parameters."""
    fov_rad = np.radians(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


def compute_ipm_homography(K, camera_height, camera_pitch_deg,
                           x_range=(-10, 10), y_range=(3, 30),
                           output_size=(400, 400)):
    """
    Compute a homography that maps image pixels to a top-down ground plane.

    Args:
        K: 3x3 intrinsic matrix
        camera_height: camera height above ground in meters
        camera_pitch_deg: camera pitch in degrees (negative = looking down)
        x_range: (min, max) lateral range in meters
        y_range: (min, max) forward range in meters
        output_size: (width, height) of the output BEV image in pixels

    Returns:
        H: 3x3 homography matrix (image -> BEV)
    """
    pitch = np.radians(camera_pitch_deg)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    x_min, x_max = x_range
    y_min, y_max = y_range

    # 4 corners of the BEV region on the ground plane (X_lateral, Y_forward, Z=0)
    ground_corners = np.array([
        [x_min, y_min, 0],  # bottom-left
        [x_max, y_min, 0],  # bottom-right
        [x_max, y_max, 0],  # top-right
        [x_min, y_max, 0],  # top-left
    ], dtype=np.float64)

    # Project ground corners to image pixels
    img_corners = []
    for gp in ground_corners:
        # In camera frame: X=lateral, Y=+down, Z=forward
        # Ground is below camera, so Y = +camera_height (positive = downward)
        p_cam = np.array([gp[0], camera_height, gp[1]])
        p_cam = R_pitch @ p_cam

        if p_cam[2] <= 0:
            raise ValueError(
                "Ground corner (%.1f, %.1f) projects behind camera. "
                "Increase y_range min or adjust pitch." % (gp[0], gp[1])
            )
        u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        img_corners.append([u, v])

    img_corners = np.float32(img_corners)

    w, h = output_size
    bev_corners = np.float32([
        [0, h],   # bottom-left
        [w, h],   # bottom-right
        [w, 0],   # top-right
        [0, 0],   # top-left
    ])

    H, _ = cv2.findHomography(img_corners, bev_corners)
    return H


def draw_bev_grid(bev_img, x_range, y_range, output_size, grid_spacing=5):
    """
    Draw a meter-scale grid overlay on the BEV image.

    Args:
        bev_img: BEV image (will be modified in-place)
        x_range: (min, max) lateral range in meters
        y_range: (min, max) forward range in meters
        output_size: (width, height) of the BEV image
        grid_spacing: meters between grid lines
    """
    w, h = output_size
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = x_max - x_min
    y_span = y_max - y_min

    color = (200, 200, 200)
    label_color = (255, 255, 255)

    # Vertical lines (lateral distance)
    x_val = math.ceil(x_min / grid_spacing) * grid_spacing
    while x_val <= x_max:
        px = int((x_val - x_min) / x_span * w)
        if 0 <= px < w:
            cv2.line(bev_img, (px, 0), (px, h), color, 1)
            label = "%+.0fm" % x_val if x_val != 0 else "0m"
            cv2.putText(bev_img, label, (px + 2, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)
        x_val += grid_spacing

    # Horizontal lines (forward distance)
    y_val = math.ceil(y_min / grid_spacing) * grid_spacing
    while y_val <= y_max:
        # y increases upward in world but downward in image
        py = int((1.0 - (y_val - y_min) / y_span) * h)
        if 0 <= py < h:
            cv2.line(bev_img, (0, py), (w, py), color, 1)
            cv2.putText(bev_img, "%.0fm" % y_val, (4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)
        y_val += grid_spacing

    # Draw vehicle position marker (bottom center)
    vx = w // 2
    vy = int((1.0 - (0 - y_min) / y_span) * h)  # y=0 is vehicle position
    if 0 <= vy < h:
        cv2.drawMarker(bev_img, (vx, vy), (0, 255, 255),
                        cv2.MARKER_DIAMOND, 12, 2)
    else:
        # Vehicle is below the BEV view, draw marker at bottom center
        cv2.drawMarker(bev_img, (vx, h - 10), (0, 255, 255),
                        cv2.MARKER_TRIANGLE_UP, 12, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SegFormer + Bird's Eye View ground projection in CARLA. "
                    "Shows camera feed and BEV segmentation side-by-side."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", metavar="NAME", default=None,
                        help="Load this map (e.g. Town01, Town02). If omitted, use current map.")
    parser.add_argument("--model",
                        default="nvidia/segformer-b0-finetuned-ade-512-512",
                        help="HuggingFace model id or local path to fine-tuned model")
    parser.add_argument("--deeplab", action="store_true",
                        help="Use DeepLabV3+ ORFD binary freespace model instead of SegFormer")
    parser.add_argument("--deeplab-model",
                        default="./training/models/deeplabv3_orfd/best_model.pth",
                        help="Path to DeepLabV3+ ORFD checkpoint (.pth)")
    parser.add_argument("--width", type=int, default=960,
                        help="Camera width (default 960, matches AC-IMX490-H120 aspect ratio)")
    parser.add_argument("--height", type=int, default=620,
                        help="Camera height (default 620, matches AC-IMX490-H120 aspect ratio)")
    parser.add_argument("--infer-every", type=int, default=1, metavar="N",
                        help="Run model every Nth frame (default 1; increase for lower-spec machines)")
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor (default 1.5)")
    parser.add_argument("--no-thread", action="store_true",
                        help="Run inference in main loop instead of background thread")
    parser.add_argument("--bev-x", type=float, default=5,
                        help="BEV lateral half-range in meters (default 5 = 10m total)")
    parser.add_argument("--bev-ymin", type=float, default=1,
                        help="BEV minimum forward distance in meters (default 1)")
    parser.add_argument("--bev-ymax", type=float, default=25,
                        help="BEV maximum forward distance in meters (default 25)")
    parser.add_argument("--bev-size", type=int, default=400,
                        help="BEV output image size in pixels (square, default 400)")
    parser.add_argument("--grid", type=float, default=5,
                        help="Grid spacing in meters (default 5; 0 to disable)")
    args = parser.parse_args()

    # ---- Connect to CARLA ----
    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Make sure CARLA is running (CarlaUE4.exe) and a map is loaded.")
        return 1

    if args.map:
        print("Loading map: %s ..." % args.map)
        client.set_timeout(120.0)
        try:
            client.load_world(args.map)
        except RuntimeError as e:
            if "not found" in str(e).lower():
                print("Map '%s' not available. Run: python scripts/list_maps.py" % args.map)
            raise

    client.set_timeout(30.0)
    world = client.get_world()
    time.sleep(2.0)

    # ---- Load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_deeplab = args.deeplab

    if use_deeplab:
        if not _HAS_SMP:
            print("segmentation-models-pytorch required for --deeplab. Install: pip install segmentation-models-pytorch")
            return 1
        print("Loading DeepLabV3+ ORFD from %s (device: %s) ..." % (args.deeplab_model, device))
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
        ckpt = torch.load(args.deeplab_model, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        print("Model loaded (epoch %d, IoU %.4f)" % (ckpt.get("epoch", -1), ckpt.get("iou", 0.0)))
        processor = None
        id2label = {0: "non-drivable", 1: "drivable"}
        is_finetuned = True
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = (60, 60, 60)    # non-drivable = dark gray (BGR)
        palette[1] = (0, 200, 0)     # drivable = green (BGR)
        palette_is_bgr = True
        deeplab_preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        print("Loading SegFormer model %s (device: %s) ..." % (args.model, device))
        print("(First run may download the model from Hugging Face.)")
        processor = SegformerImageProcessor.from_pretrained(args.model)
        model = SegformerForSemanticSegmentation.from_pretrained(args.model)
        model.to(device)
        model.eval()
        id2label = getattr(model.config, "id2label", None) or {}
        id2label = {int(k): str(v) for k, v in id2label.items()}
        is_finetuned = len(id2label) <= 30
        deeplab_preprocess = None

        if is_finetuned:
            palette = _offroad_palette_bgr(id2label)
            palette_is_bgr = True
            print("Fine-tuned model (%d classes): %s" % (len(id2label), ", ".join(id2label.values())))
        else:
            palette = _generic_palette()
            palette_is_bgr = False
            print("Pretrained model (%d classes), showing driving-relevant classes in legend." % len(id2label))

    # ---- Compute IPM homography (once) ----
    camera_height = 1.2  # matches carla.Location(x=2.8, z=1.2)
    camera_pitch = 0.0   # horizontal
    K = get_intrinsic_matrix(args.width, args.height, fov_deg=120)
    x_range = (-args.bev_x, args.bev_x)
    y_range = (args.bev_ymin, args.bev_ymax)
    bev_size = (args.bev_size, args.bev_size)

    try:
        H = compute_ipm_homography(K, camera_height, camera_pitch,
                                   x_range=x_range, y_range=y_range,
                                   output_size=bev_size)
    except ValueError as e:
        print("IPM error: %s" % e)
        return 1

    print("IPM: lateral %.0fm to %.0fm, forward %.0fm to %.0fm, BEV %dx%d px" % (
        x_range[0], x_range[1], y_range[0], y_range[1], bev_size[0], bev_size[1]))

    # ---- Spawn vehicle ----
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points on this map.")
        return 1
    vehicles = blueprint_library.filter("vehicle")
    if not vehicles:
        print("No vehicle blueprints found.")
        return 1
    vehicle_bp = blueprint_library.find("vehicle.dodge.charger_2020")
    if vehicle_bp is None:
        vehicle_bp = random.choice(vehicles)
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if vehicle is None:
        print("Spawn failed. Try again or another map.")
        return 1

    # ---- Attach camera ----
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", "120")
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=2.8, z=1.2)),
        attach_to=vehicle,
    )
    vehicle.set_autopilot(True)

    latest_frame = [None]

    def on_image(carla_image):
        latest_frame[0] = carla_image_to_bgr(carla_image)

    camera.listen(on_image)

    # ---- Display setup ----
    # Layout: [camera (W x H)] [seg (W x H)] [BEV (bev_size, resized to H)] [legend (220 x H)]
    bev_display_h = args.height
    bev_display_w = int(args.bev_size * (args.height / args.bev_size))

    window_name = "Autopilot + Ground Projection (BEV)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    total_w = int((args.width * 2 + bev_display_w + 220) * args.scale)
    total_h = int(args.height * args.scale)
    cv2.resizeWindow(window_name, total_w, total_h)
    print("Running. Press 'q' or ESC to exit.")
    if not args.no_thread:
        print("Inference runs in background thread.")

    last_seg_mask = [None]
    last_seg_bgr = [None]
    last_bev_bgr = [None]
    last_fps_timestamp = [None]
    fps_smooth = [0.0]
    stop_inference = threading.Event()

    def run_segmentation(bgr):
        """Run segmentation on a BGR frame, return colorized BGR seg image."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        if use_deeplab:
            input_tensor = deeplab_preprocess(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            seg = (mask > 0.5).astype(np.uint8)
        else:
            inputs = processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )
            seg = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        seg_colored = palette[seg % 256]
        if not palette_is_bgr:
            seg_colored = cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR)
        return seg, seg_colored

    def inference_worker():
        infer_interval = args.infer_every / 30.0
        while not stop_inference.is_set():
            stop_inference.wait(timeout=infer_interval)
            if stop_inference.is_set():
                break
            bgr = latest_frame[0]
            if bgr is None:
                continue
            bgr = bgr.copy()
            try:
                seg, seg_colored = run_segmentation(bgr)
                last_seg_mask[0] = seg
                last_seg_bgr[0] = seg_colored

                # Warp to BEV
                bev = cv2.warpPerspective(
                    seg_colored, H, bev_size,
                    flags=cv2.INTER_NEAREST,
                    borderValue=(40, 40, 40),
                )

                # Grid overlay
                if args.grid > 0:
                    draw_bev_grid(bev, x_range, y_range, bev_size, args.grid)

                last_bev_bgr[0] = bev

                t_now = time.perf_counter()
                if last_fps_timestamp[0] is not None:
                    dt = t_now - last_fps_timestamp[0]
                    if dt > 0:
                        fps_smooth[0] = 0.85 * fps_smooth[0] + 0.15 * (1.0 / dt)
                last_fps_timestamp[0] = t_now
            except Exception:
                pass

    use_thread = not args.no_thread
    if use_thread:
        worker = threading.Thread(target=inference_worker, daemon=True)
        worker.start()

    frame_count = 0
    t_last_inference = None
    try:
        while True:
            bgr = latest_frame[0]
            if bgr is None:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            h, w = bgr.shape[:2]

            if not use_thread:
                run_inference = (frame_count % args.infer_every == 0)
                if run_inference:
                    seg, seg_colored = run_segmentation(bgr)
                    last_seg_mask[0] = seg
                    last_seg_bgr[0] = seg_colored

                    bev = cv2.warpPerspective(
                        seg_colored, H, bev_size,
                        flags=cv2.INTER_NEAREST,
                        borderValue=(40, 40, 40),
                    )
                    if args.grid > 0:
                        draw_bev_grid(bev, x_range, y_range, bev_size, args.grid)

                    last_bev_bgr[0] = bev

                    t_now = time.perf_counter()
                    if t_last_inference is not None:
                        dt = t_now - t_last_inference
                        if dt > 0:
                            fps_smooth[0] = 0.85 * fps_smooth[0] + 0.15 * (1.0 / dt)
                    t_last_inference = t_now

            disp_fps = fps_smooth[0]
            bev_bgr = last_bev_bgr[0]

            seg_bgr = last_seg_bgr[0]

            if bev_bgr is not None and seg_bgr is not None:
                # Resize BEV to match camera height
                bev_resized = cv2.resize(bev_bgr, (bev_display_w, bev_display_h),
                                         interpolation=cv2.INTER_NEAREST)

                # Build legend
                legend = build_legend(id2label, palette, h, is_finetuned, palette_is_bgr)

                # Combine: [camera] [seg] [BEV] [legend]
                combined = np.hstack([bgr, seg_bgr, bev_resized, legend])

                if args.scale != 1.0:
                    new_w = int(combined.shape[1] * args.scale)
                    new_h = int(combined.shape[0] * args.scale)
                    combined = cv2.resize(combined, (new_w, new_h),
                                          interpolation=cv2.INTER_LINEAR)

                # HUD
                v = vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                cv2.putText(combined, "Inference FPS: %.1f" % disp_fps,
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Inference FPS: %.1f" % disp_fps,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh,
                            (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # BEV label (after camera + seg panels)
                bev_label_x = int(args.width * 2 * args.scale) + 10
                cv2.putText(combined, "Bird's Eye View",
                            (bev_label_x + 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Bird's Eye View",
                            (bev_label_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                cv2.imshow(window_name, combined)
            else:
                cv2.imshow(window_name, bgr)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        stop_inference.set()
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
