"""
Register all fine-tuned models into MLflow.

Usage:
    1. Start MLflow UI:  mlflow ui --port 5000
    2. Run this script:  python scripts/register_models_mlflow.py
    3. Open http://localhost:5000 to view experiments & compare models.
"""

import os
import mlflow
import mlflow.pytorch
import json

# Point at the local mlruns directory in the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "training", "models")

mlflow.set_tracking_uri(f"file:///{os.path.join(PROJECT_ROOT, 'mlruns').replace(os.sep, '/')}")

# ---------------------------------------------------------------------------
# SegFormer models
# ---------------------------------------------------------------------------
SEGFORMER_MODELS = [
    {
        "name": "rellis3d_b0_ade_arc",
        "params": {
            "dataset": "RELLIS-3D",
            "architecture": "SegFormer-B0",
            "base_pretrained": "ADE20K",
            "resolution": 256,
            "epochs": 50,
            "batch_size": 4,
            "learning_rate": 6e-5,
            "platform": "ARC (V100)",
            "precision": "fp16",
        },
        "metrics": {
            "mIoU": 0.484,
            "mean_accuracy": 0.595,
            "pixel_accuracy": 0.844,
        },
    },
    {
        "name": "rugd_b0_ade_arc",
        "params": {
            "dataset": "RUGD",
            "architecture": "SegFormer-B0",
            "base_pretrained": "ADE20K",
            "resolution": 256,
            "epochs": 50,
            "batch_size": 4,
            "learning_rate": 6e-5,
            "platform": "ARC (V100)",
            "precision": "fp16",
        },
        "metrics": {
            "mIoU": 0.341,
            "mean_accuracy": 0.438,
            "pixel_accuracy": 0.858,
        },
    },
    {
        "name": "rugd_b2_cityscapes_colab",
        "params": {
            "dataset": "RUGD",
            "architecture": "SegFormer-B2",
            "base_pretrained": "Cityscapes",
            "resolution": 512,
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 6e-5,
            "platform": "Colab (A100)",
        },
        "metrics": {
            "mIoU": 0.370,
            "mean_accuracy": 0.469,
            "pixel_accuracy": 0.872,
        },
    },
    {
        "name": "rellis3d_b2_ade_colab",
        "params": {
            "dataset": "RELLIS-3D",
            "architecture": "SegFormer-B2",
            "base_pretrained": "ADE20K",
            "resolution": 768,
            "epochs": 32,
            "batch_size": 8,
            "learning_rate": 6e-5,
            "platform": "Colab (A100)",
            "notes": "incomplete - checkpoint-26500, ran out of compute",
        },
        "metrics": {
            "mIoU": 0.447,
            "mean_accuracy": 0.572,
            "pixel_accuracy": 0.870,
        },
    },
]

# ---------------------------------------------------------------------------
# DeepLabV3+ model
# ---------------------------------------------------------------------------
DEEPLABV3_MODELS = [
    {
        "name": "deeplabv3_orfd",
        "params": {
            "dataset": "ORFD + RELLIS",
            "architecture": "DeepLabV3+ ResNet50",
            "resolution": 512,
            "task": "binary (road vs not-road)",
            "framework": "segmentation-models-pytorch",
        },
        "metrics": {
            "IoU_val": 0.858,
            "IoU_test": 0.930,
            "Dice": 0.963,
        },
    },
]


def register_segformer_models():
    """Log each SegFormer model as an MLflow run with artifacts."""
    experiment_name = "SegFormer Fine-Tuning"
    mlflow.set_experiment(experiment_name)

    for model_info in SEGFORMER_MODELS:
        model_path = os.path.join(MODELS_DIR, model_info["name"])
        if not os.path.isdir(model_path):
            print(f"  [SKIP] {model_info['name']} — folder not found")
            continue

        with mlflow.start_run(run_name=model_info["name"]):
            # Log parameters
            for k, v in model_info["params"].items():
                mlflow.log_param(k, v)

            # Log metrics
            for k, v in model_info["metrics"].items():
                mlflow.log_metric(k, v)

            # Log model artifacts (config, weights, etc.)
            mlflow.log_artifacts(model_path, artifact_path="model")

            # Log id2label mapping if present
            id2label_path = os.path.join(model_path, "id2label.json")
            if os.path.exists(id2label_path):
                with open(id2label_path) as f:
                    id2label = json.load(f)
                mlflow.log_param("num_classes", len(id2label))

            print(f"  [OK] Registered {model_info['name']}")


def register_deeplabv3_models():
    """Log each DeepLabV3+ model as an MLflow run."""
    experiment_name = "DeepLabV3+ Fine-Tuning"
    mlflow.set_experiment(experiment_name)

    for model_info in DEEPLABV3_MODELS:
        model_path = os.path.join(MODELS_DIR, model_info["name"])
        if not os.path.isdir(model_path):
            print(f"  [SKIP] {model_info['name']} — folder not found")
            continue

        with mlflow.start_run(run_name=model_info["name"]):
            for k, v in model_info["params"].items():
                mlflow.log_param(k, v)
            for k, v in model_info["metrics"].items():
                mlflow.log_metric(k, v)

            # Log the .pth file as artifact
            pth_file = os.path.join(model_path, "best_model.pth")
            if os.path.exists(pth_file):
                mlflow.log_artifact(pth_file, artifact_path="model")

            mlflow.log_param("num_classes", 1)
            print(f"  [OK] Registered {model_info['name']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Registering models in MLflow")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print("=" * 60)

    print("\n--- SegFormer models ---")
    register_segformer_models()

    print("\n--- DeepLabV3+ models ---")
    register_deeplabv3_models()

    print("\n" + "=" * 60)
    print("Done! Open http://localhost:5000 to view in MLflow UI.")
    print("To compare models: select runs → click 'Compare'")
    print("=" * 60)
