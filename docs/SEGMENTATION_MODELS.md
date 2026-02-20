# Segmentation Models

How to run pretrained and fine-tuned segmentation models in this project.

---

## Install

```powershell
pip install -r requirements-segmentation.txt
```

Models are auto-downloaded from HuggingFace on first run. No manual download needed for pretrained models.

---

## ADE20K (SegFormer) — Pretrained

150 classes (sky, road, tree, grass, earth, building, etc.). Good general-purpose baseline.

```powershell
python scripts/run_segformer_image.py path/to/image.jpg
python scripts/autopilot_segformer.py
```

### Model variants (swap via `--model`)

| Model | Speed | Accuracy |
|-------|-------|----------|
| `nvidia/segformer-b0-finetuned-ade-512-512` | Fastest (default) | Good |
| `nvidia/segformer-b2-finetuned-ade-512-512` | Medium | Better |
| `nvidia/segformer-b5-finetuned-ade-640-640` | Slowest | Best |

**Lower-spec machines:** `--width 320 --height 240 --infer-every 5`

---

## Fine-Tuned Models

See `training/models/MODEL_REGISTRY.md` for the full list with metrics. Use with any SegFormer script via `--model`:

```powershell
python scripts/autopilot_segformer.py --model training/models/rellis3d_b0_ade_arc
python scripts/autopilot_segformer.py --model training/models/rugd_b0_ade_arc
```

DeepLabV3+ (binary road detection):

```powershell
python scripts/autopilot_deeplabv3_orfd.py
```
