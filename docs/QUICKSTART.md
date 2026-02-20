# Quickstart: Zero to Running Segmentation in CARLA

Everything you need to go from a fresh machine to running fine-tuned segmentation models inside CARLA. Follow the steps in order.

---

## Step 1: Install CARLA 0.9.16

1. Download **CARLA 0.9.16** (Windows) from [CARLA GitHub Releases](https://github.com/carla-simulator/carla/releases/tag/0.9.16/).
   - Get the **pre-built Windows package** (e.g. `CARLA_0.9.16.zip`, ~20 GB).
2. Extract to `C:\CARLA_0.9.16` (or wherever you want — set `CARLA_ROOT` env var if different).
3. Verify: you should have `C:\CARLA_0.9.16\CarlaUE4.exe`.

---

## Step 2: Install Python 3.12

CARLA 0.9.16's Windows wheel requires **Python 3.12** specifically.

1. Download from [python.org](https://www.python.org/downloads/release/python-3120/).
2. During install, check **"Add Python to PATH"**.
3. Verify: `python --version` should show `3.12.x`.

---

## Step 3: Clone the Repo

```powershell
git clone https://github.com/alexhowden/carla-dev.git
cd carla-dev
```

---

## Step 4: Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

---

## Step 5: Install Dependencies

```powershell
# CARLA Python client (adjust path if your CARLA is installed elsewhere)
pip install "C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"

# Base dependencies
pip install -r requirements.txt

# Segmentation model dependencies (SegFormer, PyTorch, etc.)
pip install -r requirements-segmentation.txt

# Additional packages needed by specific scripts
pip install opencv-python pynput segmentation-models-pytorch
```

---

## Step 6: Download Model Weights from Google Drive

Some model files are too large for GitHub. Download them from the shared Google Drive:

**[Google Drive — Model Weights](https://drive.google.com/drive/u/1/folders/14IOYnkS2wluEGGEdazpp7kregTt1YTvk)**

After downloading, place the files in the correct folders:

| File | Place in |
|------|----------|
| `best_model.pth` (306 MB) | `training/models/deeplabv3_orfd/` |
| `model.safetensors` (104 MB, RUGD B2) | `training/models/rugd_b2_cityscapes_colab/` |

The B0 models (~14 MB each) are already included in the repo via `git clone`.

### Verify model files are in place

After downloading, your `training/models/` should look like:

```
training/models/
├── rellis3d_b0_ade_arc/
│   ├── config.json
│   ├── id2label.json
│   ├── label2id.json
│   ├── model.safetensors        ← included in repo (14 MB)
│   └── preprocessor_config.json
├── rugd_b0_ade_arc/
│   ├── config.json
│   ├── id2label.json
│   ├── label2id.json
│   ├── model.safetensors        ← included in repo (14 MB)
│   └── preprocessor_config.json
├── rugd_b2_cityscapes_colab/
│   ├── config.json
│   ├── id2label.json
│   ├── label2id.json
│   ├── model.safetensors        ← DOWNLOAD from Drive (104 MB)
│   └── preprocessor_config.json
├── deeplabv3_orfd/
│   └── best_model.pth           ← DOWNLOAD from Drive (306 MB)
└── MODEL_REGISTRY.md
```

---

## Step 7: Start CARLA

Open a **new terminal** (keep it running the whole time):

```powershell
cd carla-dev
.\.venv\Scripts\Activate.ps1
python scripts/start_carla.py
```

Wait for the Unreal Engine window to open and a map to load. This can take 30–90 seconds on first launch.

### Verify CARLA is running

In a **second terminal**:

```powershell
cd carla-dev
.\.venv\Scripts\Activate.ps1
python scripts/connection.py
```

You should see a successful connection message. Then try:

```powershell
python scripts/connection.py --demo
```

This spawns a vehicle with a camera — you should see a window pop up.

---

## Step 8: Run Segmentation Models

With CARLA still running in the first terminal, run any of these in the second terminal:

### SegFormer (multi-class terrain segmentation)

```powershell
# ADE20K pretrained (150 classes, auto-downloaded from HuggingFace)
python scripts/autopilot_segformer.py

# Fine-tuned on RELLIS-3D (21 off-road classes)
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc

# Fine-tuned on RUGD (24 off-road classes)
python scripts/autopilot_segformer.py --model training/models/rugd_b0_ade_arc

# Fine-tuned RUGD B2 (larger model, need Drive download)
python scripts/autopilot_segformer.py --model training/models/rugd_b2_cityscapes_colab
```

### DeepLabV3+ (binary road detection)

```powershell
python scripts/autopilot_deeplabv3_orfd.py
```

Green overlay = drivable road. Requires `best_model.pth` from Drive.

### Ground Projection (Bird's Eye View)

```powershell
python scripts/autopilot_ground_projection.py --model training/models/rellis3d_b0_ade_arc
```

Shows camera + segmentation + top-down BEV with meter grid side-by-side.

### Manual Drive (WASD)

Any of the above also have manual-drive versions:

```powershell
python scripts/manual_control_segformer.py --model training/models/rellis3d_b0_ade_arc
python scripts/manual_control_ground_projection.py --model training/models/rellis3d_b0_ade_arc
python scripts/manual_control_deeplabv3_orfd.py
```

Controls: **W** = throttle, **S** = brake, **A** = left, **D** = right, **SPACE** = hand brake, **Q** = toggle reverse. Press **q** or **ESC** to exit.

---

## Step 9: Load a Different Map (Optional)

```powershell
# See available maps
python scripts/list_maps.py

# Load a specific map
python scripts/load_map.py --map Town03

# Then run any script — it uses the current map
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc
```

Best maps for off-road: **Town07** (rural, requires additional maps package) or **Town03** (mixed terrain, included in base install).

---

## Performance Tips

If inference is slow (no GPU, or weak GPU):

```powershell
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc --width 480 --height 310 --infer-every 5
```

| Flag | Effect |
|------|--------|
| `--width 480 --height 310` | Half resolution (faster) |
| `--infer-every 5` | Run model every 5th frame |
| `--no-thread` | Run inference in main loop |

All models work on CPU — just slower.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ImportError: No module named 'carla'` | Install the CARLA wheel (Step 5). Path must match your CARLA install. |
| `python --version` is not 3.12 | Install Python 3.12 and make sure your venv uses it. |
| Connection failed | Make sure CARLA is running (Step 7) and ports 2000/2001 are free. |
| `Map 'Town07' not found` | Use `--map Town03` instead, or install CARLA's additional maps package. |
| `No module named 'smp'` | Run `pip install segmentation-models-pytorch`. |
| Model loads but no output / crash | Make sure you downloaded the weight files from Drive (Step 6). |
| Slow / low FPS | Use performance flags (see above). B0 models are fastest. |

---

## What's Next

- **[CARLA_AND_PERCEPTION_GUIDE.md](CARLA_AND_PERCEPTION_GUIDE.md)** — full script reference, camera params, map details
- **[USING_MODELS.md](USING_MODELS.md)** — model details and quick reference table
- **[GROUND_PROJECTION.md](GROUND_PROJECTION.md)** — how the bird's eye view works
- **[GREAT_LAKES_TRAINING.md](GREAT_LAKES_TRAINING.md)** — how to fine-tune new models on Great Lakes HPC
- **`training/models/MODEL_REGISTRY.md`** — all model metrics and comparison
