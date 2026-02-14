# Using the Fine-Tuned Models

How to load and use the fine-tuned SegFormer and DeepLabV3+ models for testing in CARLA and evaluating on datasets.

---

## 1. Available Models

See `training/models/MODEL_REGISTRY.md` for the full list with metrics. Summary:

| Folder | Type | Dataset | Classes | Task |
|--------|------|---------|---------|------|
| `rellis3d_b0_ade_arc` | SegFormer B0 | RELLIS-3D | 21 | Multi-class terrain |
| `rugd_b0_ade_arc` | SegFormer B0 | RUGD | 24 | Multi-class terrain |
| `rugd_b2_cityscapes_colab` | SegFormer B2 | RUGD | 24 | Multi-class terrain |
| `deeplabv3_orfd` | DeepLabV3+ ResNet50 | ORFD | 2 | Binary road detection |

---

## 2. Setup

### Clone the repo

```bash
git clone https://github.com/alexhowden/carla-dev.git
cd carla-dev
```

### Install dependencies

```bash
pip install -r requirements-segmentation.txt
pip install opencv-python pynput
```

### Download large model files

Some model weight files are too large for GitHub and must be downloaded separately from Google Drive.

**Files NOT in the repo (must download manually):**

| File | Size | Google Drive location |
|------|------|---------------------|
| `training/models/deeplabv3_orfd/best_model.pth` | 306 MB | Ask Alex for the shared Drive link |
| `training/models/rugd_b2_cityscapes_colab/model.safetensors` | 104 MB | Ask Alex for the shared Drive link |

> **Note:** The Google Drive folder will be shared with the team once models are uploaded. If you don't have access yet, ask Alex.

**Files already in the repo (come with `git clone`):**

| File | Size |
|------|------|
| `training/models/rellis3d_b0_ade_arc/model.safetensors` | 14 MB |
| `training/models/rugd_b0_ade_arc/model.safetensors` | 14 MB |
| All `config.json`, `id2label.json`, `preprocessor_config.json` | tiny |

After downloading, place the files in the matching `training/models/` subfolders.

### GPU memory requirements

| Model | GPU VRAM needed | CPU-only? |
|-------|----------------|----------|
| SegFormer B0 (14 MB) | ~1 GB | Yes, but slow (~1-2 FPS) |
| SegFormer B2 (104 MB) | ~3 GB | Yes, but very slow (<0.5 FPS) |
| DeepLabV3+ ResNet50 (306 MB) | ~2 GB | Yes, but slow |

All models run on CPU if no GPU is available — just slower. Use `--infer-every 5 --width 480 --height 310` to improve FPS on CPU.

---

## 3. Testing in CARLA

### Prerequisites

- CARLA simulator running (`python scripts/start_carla.py` or launch `CarlaUE4.exe`)
- A map loaded (e.g. `python scripts/load_map.py --map Town02`)

### SegFormer models (autopilot)

```bash
# ADE20K pretrained (default, no download needed — fetched from HuggingFace)
python scripts/autopilot_segformer.py

# Fine-tuned on RELLIS-3D (B0)
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc

# Fine-tuned on RUGD (B0)
python scripts/autopilot_segformer.py --model training/models/rugd_b0_ade_arc

# Fine-tuned on RUGD (B2 — need to download model.safetensors first)
python scripts/autopilot_segformer.py --model training/models/rugd_b2_cityscapes_colab
```

### SegFormer models (manual drive)

Same `--model` flag, but you drive with WASD:

```bash
python scripts/manual_control_segformer.py --model training/models/rellis3d_b0_ade_arc
```

Controls: **W** = throttle, **S** = brake, **A** = left, **D** = right, **SPACE** = hand brake, **Q** = toggle reverse.

### DeepLabV3+ model (autopilot)

```bash
python scripts/autopilot_deeplabv3_orfd.py
# or specify path explicitly:
python scripts/autopilot_deeplabv3_orfd.py --model training/models/deeplabv3_orfd/best_model.pth
```

### DeepLabV3+ model (manual drive)

```bash
python scripts/manual_control_deeplabv3_orfd.py
```

### Performance options

For lower-spec machines (no GPU, MacBook, etc.):

```bash
# Lower resolution + skip frames
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc \
    --width 480 --height 310 --infer-every 5
```

| Flag | Effect |
|------|--------|
| `--width 480 --height 310` | Half resolution (faster inference) |
| `--infer-every 5` | Run model every 5th frame (reuse last result) |
| `--no-thread` | Run inference in main loop (default: background thread) |

---

## 4. Loading Models in Python (Outside CARLA)

### SegFormer (HuggingFace)

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import torch

# Load model and processor
model_path = "training/models/rellis3d_b0_ade_arc"
processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path)
model.eval()

# Run inference on an image
image = Image.open("test_image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get predicted class per pixel
logits = outputs.logits  # shape: (1, num_classes, H/4, W/4)
logits = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # shape: (H, W)

# Map class IDs to names
id2label = model.config.id2label
for class_id in set(pred.flatten()):
    print(f"  Class {class_id}: {id2label[str(class_id)]}")
```

### DeepLabV3+ (PyTorch)

```python
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

# Load model
model_path = "training/models/deeplabv3_orfd/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try loading as full model first
model = torch.load(model_path, map_location=device, weights_only=False)
if hasattr(model, "eval"):
    model.eval()
else:
    # If it's a state_dict, load into architecture
    from torchvision.models.segmentation import deeplabv3_resnet50
    arch = deeplabv3_resnet50(num_classes=2, weights_backbone=None)
    arch.load_state_dict(model, strict=False)
    model = arch
    model.to(device)
    model.eval()

# Preprocess (ImageNet normalization)
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("test_image.jpg").convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Handle output format
if isinstance(output, dict):
    logits = output["out"]
else:
    logits = output

# Binary mask: class 1 = road
logits = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
if logits.shape[1] >= 2:
    road_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
else:
    road_mask = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)

# road_mask: 1 = road, 0 = not road
print(f"Road pixels: {road_mask.sum()} / {road_mask.size} ({100*road_mask.mean():.1f}%)")
```

---

## 5. Evaluating on a Test Set

### SegFormer evaluation

Compute mIoU, mean accuracy, and pixel accuracy on a dataset.

> **Dataset folder structure:** The scripts below assume images and labels are in separate folders with matching filenames (e.g. `images/test/001.jpg` and `labels/test/001.png`). Adjust the paths to match your dataset layout. See `training/models/MODEL_REGISTRY.md` for dataset details.

```python
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
from pathlib import Path

model_path = "training/models/rellis3d_b0_ade_arc"
processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_classes = len(model.config.id2label)
id2label = model.config.id2label
ignore_index = 255

# Paths to test images and labels
# Adjust these to match your dataset structure
image_dir = Path("datasets/rellis3d/images/test")
label_dir = Path("datasets/rellis3d/labels/test")

confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

for img_path in sorted(image_dir.glob("*.jpg")):
    label_path = label_dir / img_path.name.replace(".jpg", ".png")
    if not label_path.exists():
        continue

    image = Image.open(img_path).convert("RGB")
    label = np.array(Image.open(label_path))

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    logits = torch.nn.functional.interpolate(
        logits, size=label.shape, mode="bilinear", align_corners=False
    )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Update confusion matrix (ignore void pixels)
    valid = label != ignore_index
    for gt_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion[gt_class, pred_class] += np.sum(
                (label[valid] == gt_class) & (pred[valid] == pred_class)
            )

# Compute metrics from confusion matrix
iou_per_class = np.zeros(num_classes)
acc_per_class = np.zeros(num_classes)
for c in range(num_classes):
    tp = confusion[c, c]
    fp = confusion[:, c].sum() - tp
    fn = confusion[c, :].sum() - tp
    iou_per_class[c] = tp / (tp + fp + fn + 1e-10)
    acc_per_class[c] = tp / (confusion[c, :].sum() + 1e-10)

pixel_acc = np.diag(confusion).sum() / (confusion.sum() + 1e-10)
mean_iou = np.nanmean(iou_per_class)
mean_acc = np.nanmean(acc_per_class)

print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"Mean Accuracy:  {mean_acc:.4f}")
print(f"Mean IoU:       {mean_iou:.4f}")
print()
print("Per-class IoU:")
for c in range(num_classes):
    print(f"  {id2label[str(c)]:20s}  IoU={iou_per_class[c]:.4f}  Acc={acc_per_class[c]:.4f}")
```

### DeepLabV3+ evaluation (binary)

```python
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Load model (same as Section 4)
model = torch.load("training/models/deeplabv3_orfd/best_model.pth",
                    map_location="cpu", weights_only=False)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Adjust paths to your test set
image_dir = Path("datasets/orfd/images/test")
label_dir = Path("datasets/orfd/labels/test")

tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0

for img_path in sorted(image_dir.glob("*.jpg")):
    label_path = label_dir / img_path.name.replace(".jpg", ".png")
    if not label_path.exists():
        continue

    image = Image.open(img_path).convert("RGB")
    label = np.array(Image.open(label_path))
    label = (label > 0).astype(np.uint8)  # binarize: 1 = road

    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    logits = output["out"] if isinstance(output, dict) else output
    logits = torch.nn.functional.interpolate(
        logits, size=label.shape, mode="bilinear", align_corners=False
    )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    tp_total += np.sum((pred == 1) & (label == 1))
    fp_total += np.sum((pred == 1) & (label == 0))
    fn_total += np.sum((pred == 0) & (label == 1))
    tn_total += np.sum((pred == 0) & (label == 0))

iou = tp_total / (tp_total + fp_total + fn_total + 1e-10)
dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-10)
precision = tp_total / (tp_total + fp_total + 1e-10)
recall = tp_total / (tp_total + fn_total + 1e-10)
accuracy = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total + 1e-10)

print(f"IoU:       {iou:.4f}")
print(f"Dice:      {dice:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
```

---

## 6. Quick Reference

| Task | Command |
|------|---------|
| Test SegFormer in CARLA (autopilot) | `python scripts/autopilot_segformer.py --model training/models/<folder>` |
| Test SegFormer in CARLA (manual) | `python scripts/manual_control_segformer.py --model training/models/<folder>` |
| Test DeepLabV3+ in CARLA (autopilot) | `python scripts/autopilot_deeplabv3_orfd.py` |
| Test DeepLabV3+ in CARLA (manual) | `python scripts/manual_control_deeplabv3_orfd.py` |
| Lower-spec mode | Add `--width 480 --height 310 --infer-every 5` |
| Evaluate SegFormer on test set | See Section 5 (Python script) |
| Evaluate DeepLabV3+ on test set | See Section 5 (Python script) |
