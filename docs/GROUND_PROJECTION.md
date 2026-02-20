# Ground Projection (Bird's Eye View)

Converts a segmentation mask from camera perspective to a top-down ground plane using Inverse Perspective Mapping (IPM). The planner needs traversable terrain in **world coordinates**, not pixel coordinates.

---

## 1. What Is Ground Projection?

IPM warps the camera image so the ground plane appears as if viewed from directly above, with uniform scale (each pixel = known meters).

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

**Limitation:** IPM assumes flat ground. Works well for roads but distorts anything above the ground plane (trees, poles, buildings). On slopes the projection is approximate.

---

## 2. How It Works

The implementation is in `autopilot_ground_projection.py` and `manual_control_ground_projection.py`. The key steps:

1. **Intrinsic matrix K** — computed from CARLA camera FOV + resolution (no calibration needed in sim)
2. **Project 4 ground corners** through the camera model to get image pixel positions
3. **Compute homography H** via `cv2.findHomography()` (image pixels → BEV pixels)
4. **Warp segmentation mask** with `cv2.warpPerspective(seg_colored, H, bev_size)`

The homography is computed **once** at startup (camera doesn't move relative to the vehicle).

### Current CARLA parameters

| Parameter | Value |
|-----------|-------|
| FOV | 120° (matches AC-IMX490-H120) |
| Resolution | 960×620 |
| Camera height | 1.2m above ground |
| Camera pitch | 0° (horizontal) |
| Camera mount | x=2.8m forward, z=1.2m up on vehicle |
| BEV range | ±5m lateral, 1–25m forward (configurable via `--bev-x`, `--bev-ymin`, `--bev-ymax`) |

---

## 3. Real Camera (AC-IMX490-H120)

In CARLA we have perfect camera parameters. On the real camera:

| Parameter | CARLA | Real Camera |
|-----------|-------|-------------|
| Intrinsics (K) | Computed from FOV | Checkerboard calibration needed |
| Distortion | None (pinhole) | Must undistort first (120° lens has barrel distortion) |
| Extrinsics | Exact from API | Measure height + pitch manually |
| Ground plane | Perfectly flat | Approximate (slopes, bumps) |
| Resolution | 960×620 (configurable) | 2880×1860 (native) |

Steps for real camera: calibrate with checkerboard → undistort frames → measure camera mount height/pitch → use same `compute_ipm_homography()` with calibrated K.

See [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for the calibration procedure.

---

## 4. Path Planning Notes

- **BEV mask = costmap.** Traversable classes = low cost, obstacles = high cost, void = unknown.
- **Each BEV pixel = known distance** (determined by range and output size).
- **Horizon region is low-resolution** — one pixel covers many meters. Trust the lower portion more.

---

## 5. References

- [CARLA sensor reference — RGB camera](https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera)
- [Inverse Perspective Mapping (IPM)](https://en.wikipedia.org/wiki/Inverse_perspective_mapping)
