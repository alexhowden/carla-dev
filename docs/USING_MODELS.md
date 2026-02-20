# Using the Fine-Tuned Models

---

## 1. Available Models

See `training/models/MODEL_REGISTRY.md` for full metrics.

| Folder | Type | Dataset | Task |
|--------|------|---------|------|
| `rellis3d_b0_ade_arc` | SegFormer B0 | RELLIS-3D | Multi-class terrain |
| `rugd_b0_ade_arc` | SegFormer B0 | RUGD | Multi-class terrain |
| `rugd_b2_cityscapes_colab` | SegFormer B2 | RUGD | Multi-class terrain |
| `deeplabv3_orfd` | DeepLabV3+ ResNet50 | ORFD | Binary road detection |

---

## 2. Setup

```bash
git clone https://github.com/alexhowden/carla-dev.git
cd carla-dev
pip install -r requirements-segmentation.txt
pip install opencv-python pynput
```

### Large model files (download from Google Drive)

| File | Size |
|------|------|
| `training/models/deeplabv3_orfd/best_model.pth` | 306 MB |
| `training/models/rugd_b2_cityscapes_colab/model.safetensors` | 104 MB |

B0 models (14 MB each) are included in the repo. Download larger files from the [shared Google Drive](https://drive.google.com/drive/u/1/folders/14IOYnkS2wluEGGEdazpp7kregTt1YTvk).

---

## 3. Testing in CARLA

CARLA must be running first: `python scripts/start_carla.py`

### SegFormer

```bash
python scripts/autopilot_segformer.py
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc
python scripts/autopilot_segformer.py --model training/models/rugd_b0_ade_arc
python scripts/manual_control_segformer.py --model training/models/rellis3d_b0_ade_arc
```

### DeepLabV3+

```bash
python scripts/autopilot_deeplabv3_orfd.py
python scripts/manual_control_deeplabv3_orfd.py
```

### Ground projection (BEV)

```bash
python scripts/autopilot_ground_projection.py --model training/models/rellis3d_b0_ade_arc
python scripts/manual_control_ground_projection.py --model training/models/rellis3d_b0_ade_arc
```

### Performance options

| Flag | Effect |
|------|--------|
| `--width 480 --height 310` | Half resolution (faster) |
| `--infer-every 5` | Run model every 5th frame |
| `--no-thread` | Run inference in main loop |

All models run on CPU if no GPU — just slower.

---

## 4. Quick Reference

| Task | Command |
|------|---------|
| SegFormer autopilot | `python scripts/autopilot_segformer.py --model training/models/<folder>` |
| SegFormer manual | `python scripts/manual_control_segformer.py --model training/models/<folder>` |
| DeepLabV3+ autopilot | `python scripts/autopilot_deeplabv3_orfd.py` |
| DeepLabV3+ manual | `python scripts/manual_control_deeplabv3_orfd.py` |
| BEV autopilot | `python scripts/autopilot_ground_projection.py --model training/models/<folder>` |
| BEV manual | `python scripts/manual_control_ground_projection.py --model training/models/<folder>` |
| Lower-spec mode | Add `--width 480 --height 310 --infer-every 5` |
