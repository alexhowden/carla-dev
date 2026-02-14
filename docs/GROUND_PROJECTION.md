# Ground Projection (Bird's Eye View)

How to project a semantic segmentation mask from camera image space to a top-down ground plane (bird's eye view). This is essential for path planning — the planner needs to know where traversable terrain is in **world coordinates**, not pixel coordinates.

---

## 1. What Is Ground Projection?

A camera image is a 2D projection of a 3D scene. Objects farther away appear smaller and closer together (perspective). **Ground projection** (also called Inverse Perspective Mapping / IPM) reverses this: it warps the image so that the ground plane appears as if viewed from directly above.

```
Camera view (perspective)          Bird's eye view (top-down)
┌─────────────────────┐            ┌─────────────────────┐
│         sky         │            │                     │
│    ───road───       │    ──▶     │     road (flat)     │
│  ────road────       │            │     road (flat)     │
│ ─────road─────      │            │     road (flat)     │
└─────────────────────┘            └─────────────────────┘
  (near pixels = big)               (uniform scale)
```

### Important limitation

**IPM assumes the ground is flat.** It works well for roads and flat terrain but distorts anything above the ground plane (trees, vehicles, poles, buildings). For off-road terrain with slopes, the projection will be approximate. This is acceptable for initial path planning but should be noted.

---

## 2. The Math

### Camera model

A pinhole camera projects a 3D world point **P** = (X, Y, Z) to a 2D pixel (u, v) via:

```
s * [u, v, 1]ᵀ = K * [R | t] * [X, Y, Z, 1]ᵀ
```

Where:
- **K** = 3×3 intrinsic matrix (focal length, principal point)
- **[R | t]** = 3×4 extrinsic matrix (camera rotation and translation in world frame)
- **s** = scale factor

### Intrinsic matrix K

```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```

- **fx, fy** = focal lengths in pixels
- **cx, cy** = principal point (usually image center)

For a camera with horizontal FOV and image width W:

```
fx = W / (2 * tan(HFOV / 2))
```

### Ground plane assumption

If we assume the ground is at Z = 0 (or Y = 0, depending on coordinate convention), the 3D-to-2D mapping reduces to a **3×3 homography matrix H** that maps ground points to image pixels:

```
s * [u, v, 1]ᵀ = H * [X_ground, Y_ground, 1]ᵀ
```

The **inverse** H⁻¹ maps image pixels back to ground coordinates — this is the ground projection.

### Computing H from four point correspondences

The simplest approach: pick 4 points in the image whose ground-plane positions you know, then compute the homography:

```python
import cv2
import numpy as np

# 4 points in image (u, v) — e.g. corners of a known rectangle on the ground
img_pts = np.float32([[200, 400], [760, 400], [100, 600], [860, 600]])

# Corresponding points in ground plane (X, Y) in meters
ground_pts = np.float32([[-2, 10], [2, 10], [-3, 5], [3, 5]])

# Compute homography
H, _ = cv2.findHomography(img_pts, ground_pts)

# Warp the segmentation mask to bird's eye view
bev = cv2.warpPerspective(seg_mask, H, (output_width, output_height))
```

---

## 3. Ground Projection in CARLA

CARLA gives us **perfect camera parameters** — no calibration needed.

### 3a. Get camera intrinsics from CARLA

```python
import numpy as np

def get_intrinsic_matrix(width, height, fov_deg):
    """Compute the camera intrinsic matrix from CARLA camera parameters."""
    fov_rad = np.radians(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    return K

# Our current CARLA camera settings (AC-IMX490-H120 match)
K = get_intrinsic_matrix(width=960, height=620, fov_deg=120)
```

### 3b. Get camera extrinsics from CARLA

CARLA provides the exact camera transform (position + rotation) in world coordinates:

```python
def get_extrinsic_matrix(camera_transform):
    """Build the 4x4 extrinsic matrix from a CARLA transform."""
    # CARLA uses left-handed coordinate system (UE4):
    #   X = forward, Y = right, Z = up
    # Standard camera convention:
    #   X = right, Y = down, Z = forward
    loc = camera_transform.location
    rot = camera_transform.rotation

    # Convert CARLA rotation (degrees) to rotation matrix
    pitch = np.radians(rot.pitch)
    yaw   = np.radians(rot.yaw)
    roll  = np.radians(rot.roll)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])

    R = Ry @ Rx @ Rz  # CARLA rotation order: yaw, pitch, roll

    t = np.array([loc.x, loc.y, loc.z])

    # 4x4 extrinsic matrix (world to camera)
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ t
    return E
```

### 3c. Full IPM pipeline in CARLA

```python
import cv2
import numpy as np

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
        H_inv: 3x3 homography matrix (image → BEV)
    """
    pitch = np.radians(camera_pitch_deg)

    # Rotation matrix for camera pitch (around X axis)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Ground plane normal in camera frame: n = R_pitch @ [0, 1, 0]
    # Distance from camera to ground: d = camera_height

    # Define 4 corners of the BEV region in ground coordinates (X_right, Y_forward)
    x_min, x_max = x_range
    y_min, y_max = y_range

    ground_corners = np.array([
        [x_min, y_min, 0],  # bottom-left
        [x_max, y_min, 0],  # bottom-right
        [x_max, y_max, 0],  # top-right
        [x_min, y_max, 0],  # top-left
    ], dtype=np.float64)

    # Project ground corners to image pixels
    # In camera frame: X_cam = X_ground, Y_cam = -camera_height, Z_cam = Y_ground
    # (assuming camera is at height looking forward with pitch)
    img_corners = []
    for gp in ground_corners:
        # Transform ground point to camera frame
        p_cam = np.array([gp[0], -camera_height, gp[1]])  # X=lateral, Y=up, Z=forward
        p_cam = R_pitch @ p_cam  # apply pitch

        # Project to image
        if p_cam[2] <= 0:
            continue  # behind camera
        u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        img_corners.append([u, v])

    if len(img_corners) != 4:
        raise ValueError("Some ground corners project behind the camera. Adjust y_range.")

    img_corners = np.float32(img_corners)

    # BEV output corners (pixel coordinates in output image)
    w, h = output_size
    bev_corners = np.float32([
        [0, h],      # bottom-left
        [w, h],      # bottom-right
        [w, 0],      # top-right
        [0, 0],      # top-left
    ])

    # Compute homography: image → BEV
    H, _ = cv2.findHomography(img_corners, bev_corners)
    return H


def apply_ground_projection(seg_mask, H, output_size=(400, 400)):
    """
    Warp a segmentation mask to bird's eye view.

    Args:
        seg_mask: HxW numpy array (class IDs or binary mask)
        H: 3x3 homography from compute_ipm_homography()
        output_size: (width, height) of output

    Returns:
        bev: warped top-down view
    """
    w, h = output_size
    bev = cv2.warpPerspective(
        seg_mask, H, (w, h),
        flags=cv2.INTER_NEAREST,  # INTER_NEAREST preserves class IDs
        borderValue=255,           # fill unknown areas with void
    )
    return bev
```

### 3d. Example usage in a CARLA script

```python
# Camera parameters (match our CARLA scripts)
K = get_intrinsic_matrix(width=960, height=620, fov_deg=120)

# Camera is mounted at x=2.8, z=1.2 on the vehicle (see scripts)
camera_height = 1.2  # meters above ground
camera_pitch = 0.0   # degrees (0 = horizontal, negative = looking down)

# Compute homography once (doesn't change unless camera moves)
H = compute_ipm_homography(
    K, camera_height, camera_pitch,
    x_range=(-8, 8),    # 16m lateral view
    y_range=(3, 25),     # 3m to 25m ahead
    output_size=(400, 400),
)

# In the main loop, after getting segmentation mask:
bev_mask = apply_ground_projection(seg_mask, H, output_size=(400, 400))

# Colorize and display
bev_colored = palette[bev_mask]
cv2.imshow("Bird's Eye View", bev_colored)
```

---

## 4. Ground Projection with the Real Camera (AC-IMX490-H120)

With the real camera, you don't have perfect parameters — you need to calibrate.

### 4a. Camera calibration

Use a checkerboard pattern to get intrinsics and distortion coefficients:

```python
import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners)
ROWS = 6
COLS = 9
SQUARE_SIZE = 0.025  # meters (size of each square)

objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_SIZE

obj_points = []
img_points = []

images = glob.glob("calibration_images/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)

# Calibrate
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("Intrinsic matrix K:")
print(K)
print("Distortion coefficients:", dist_coeffs)

# Save for later use
np.savez("camera_calibration.npz", K=K, dist=dist_coeffs)
```

### 4b. Undistort before projection

The AC-IMX490-H120 has a 120° FOV lens — there **will** be barrel distortion. Undistort the image before running segmentation or ground projection:

```python
# Load calibration
calib = np.load("camera_calibration.npz")
K = calib["K"]
dist = calib["dist"]

# Compute undistortion maps once
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (2880, 1860), cv2.CV_32FC1)

# In the main loop:
undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
```

### 4c. Measure camera extrinsics on the vehicle

On the real vehicle, you need to measure:

1. **Camera height** above the ground (in meters)
2. **Camera pitch** angle (degrees below horizontal)
3. **Camera position** relative to the vehicle center (lateral and longitudinal offset)

These can be measured with a tape measure and inclinometer, or estimated from known reference points in the scene.

Then use the same `compute_ipm_homography()` function from Section 3c with the real K matrix and measured height/pitch.

### 4d. Differences from CARLA

| Parameter | CARLA | Real Camera |
|-----------|-------|-------------|
| Intrinsics (K) | Computed from FOV | From checkerboard calibration |
| Distortion | None (pinhole) | Must undistort first |
| Extrinsics | Exact from API | Measured manually |
| Ground plane | Perfectly flat | Approximate (slopes, bumps) |
| Resolution | 960×620 (configurable) | 2880×1860 (native) |

---

## 5. Tips for Path Planning Integration

- **Use the BEV segmentation mask directly** as a costmap for the planner. Traversable classes (road, dirt, grass) = low cost, obstacles (tree, rock, vehicle) = high cost, void = unknown.
- **Scale matters:** each pixel in the BEV image corresponds to a known real-world distance (determined by `x_range`, `y_range`, and `output_size`). For a 400×400 BEV with x_range=(-8,8) and y_range=(3,25): each pixel = 4cm × 5.5cm.
- **Only trust the lower portion of the image.** The upper part of the image (near the horizon) maps to very distant ground where a single pixel covers many meters — resolution is poor and errors are amplified.
- **Temporal smoothing:** average BEV masks over 2-3 frames to reduce noise from segmentation flickering.
- **Off-road caveat:** IPM assumes flat ground. On slopes, the projection will stretch/compress. For rough terrain, consider using depth estimation (monocular or stereo) instead of pure IPM for more accurate 3D mapping.

---

## 6. References

- [CARLA sensor reference — RGB camera attributes](https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera)
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Inverse Perspective Mapping (IPM) explained](https://en.wikipedia.org/wiki/Inverse_perspective_mapping)
- Bertozzi & Broggi, "GOLD: A Parallel Real-Time Stereo Vision System for Generic Obstacle and Lane Detection" (classic IPM paper)
