# MLflow Setup Guide

Set up MLflow to track training experiments, compare model performance, and manage model versions — replacing the manual `MODEL_REGISTRY.md` workflow.

---

## 1. Overview

MLflow has three main components we'll use:

- **Tracking** — auto-log hyperparameters, metrics (mIoU, loss, etc.), and artifacts for every training run
- **Model Registry** — version trained models centrally (v1, v2, v3) with stage labels (staging, production)
- **UI Dashboard** — web interface to compare runs side-by-side, view loss curves, filter by dataset/model size

### Architecture for our team

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Colab (A100) │     │  ARC / Great  │     │  Local (Mac   │
│  Training     │────▶│  Lakes HPC   │────▶│  or Windows)  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │         log runs + artifacts            │
       └────────────┬───────┘────────────────────┘
                    ▼
           ┌────────────────┐
           │  MLflow Server  │  (local or shared VM)
           │  + Artifact     │
           │    Store        │
           └────────────────┘
```

---

## 2. Installation

```bash
pip install mlflow
```

Add to `requirements-segmentation.txt` if not already there.

For Colab notebooks, add to the install cell:

```python
!pip install mlflow
```

---

## 3. Start the MLflow Server (Local)

The simplest setup — run on your own machine:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
```

This starts:
- **Tracking server** at `http://localhost:5000`
- **Backend store** in a local SQLite database (`mlflow.db`)
- **Artifact store** in `./mlflow-artifacts/` (where model files get saved)

Open `http://localhost:5000` in your browser to see the dashboard.

### Running as a background process

```bash
# Start in background
nohup mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts &

# Check it's running
curl http://localhost:5000/health
```

---

## 4. Integrate with Training Scripts

### 4a. SegFormer training (HuggingFace Trainer)

Add to `training/train_segformer.py` or the Colab notebook:

```python
import mlflow
import mlflow.pytorch

# Point to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # or your server's IP
mlflow.set_experiment("segformer-finetuning")

# Start a run
with mlflow.start_run(run_name="rellis3d_b2_ade_768"):
    # Log hyperparameters
    mlflow.log_params({
        "dataset": "rellis3d",
        "model_size": "b2",
        "base_pretrained": "ade20k",
        "image_size": 768,
        "batch_size": 4,
        "learning_rate": 6e-5,
        "epochs": 50,
        "platform": "colab_a100",
    })

    # ... run trainer.train() ...

    # Log final metrics
    mlflow.log_metrics({
        "mIoU": 0.484,
        "mean_acc": 0.595,
        "pixel_acc": 0.844,
    })

    # Log the trained model as an artifact
    mlflow.pytorch.log_model(model, "segformer-model")

    # Or log a directory of files
    mlflow.log_artifacts("path/to/model/folder", artifact_path="model")
```

### 4b. HuggingFace Trainer auto-logging (easiest)

MLflow has built-in HuggingFace integration. Just set an environment variable:

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "segformer-finetuning"

# Enable HuggingFace auto-logging
import mlflow
mlflow.autolog()

# Now just run trainer.train() as normal — MLflow logs everything automatically:
# - All TrainingArguments as params
# - Loss, learning rate per step
# - Eval metrics per epoch
# - Model checkpoints as artifacts
trainer.train()
```

This is the **recommended approach** — zero code changes to the training loop.

### 4c. DeepLabV3+ training (raw PyTorch)

For non-HuggingFace training loops:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("deeplabv3-finetuning")

with mlflow.start_run(run_name="deeplabv3_orfd"):
    mlflow.log_params({
        "dataset": "orfd",
        "architecture": "deeplabv3_resnet50",
        "input_size": 512,
        "num_classes": 2,
    })

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_iou = evaluate(...)

        # Log metrics per epoch
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_iou": val_iou,
        }, step=epoch)

    # Log final model
    mlflow.pytorch.log_model(model, "deeplabv3-model")
```

---

## 5. Using MLflow from Colab

Colab can't reach `localhost:5000` on your Mac. Two options:

### Option A: Log locally on Colab, download later

```python
# No server needed — logs to a local directory
mlflow.set_tracking_uri("file:///content/drive/MyDrive/mlflow-runs")
```

This saves all run data to Google Drive. You can view it later by running `mlflow ui` pointed at that directory:

```bash
# On your local machine, after syncing the folder from Drive:
mlflow ui --backend-store-uri ./mlflow-runs
```

**This is the simplest approach and recommended for now.**

### Option B: Expose local server via ngrok

```bash
# On your Mac:
pip install pyngrok
ngrok http 5000
# Gives you a URL like https://abc123.ngrok.io
```

```python
# In Colab:
mlflow.set_tracking_uri("https://abc123.ngrok.io")
```

Works but requires ngrok running the entire training session. Fragile.

---

## 6. Viewing and Comparing Runs

```bash
# Start the UI (if not already running as a server)
mlflow ui --port 5000
```

In the dashboard you can:
- **Compare runs** — select multiple runs and click "Compare"
- **View loss curves** — click a run → Metrics tab
- **Filter** — by experiment, parameter values, metric thresholds
- **Download artifacts** — model files, configs, etc.

### Useful CLI commands

```bash
# List experiments
mlflow experiments search

# List runs in an experiment
mlflow runs list --experiment-id 1

# Download a model artifact
mlflow artifacts download --run-id <RUN_ID> --artifact-path model -d ./downloaded_model
```

---

## 7. Model Registry (Optional, for later)

Once you have multiple good models, use the registry to track which is "production":

```python
# Register a model from a run
mlflow.register_model("runs:/<RUN_ID>/segformer-model", "segformer-rellis3d")

# Transition to production
from mlflow import MlflowClient
client = MlflowClient()
client.transition_model_version_stage("segformer-rellis3d", version=2, stage="Production")
```

Then load the production model anywhere:

```python
import mlflow.pytorch
model = mlflow.pytorch.load_model("models:/segformer-rellis3d/Production")
```

---

## 8. Recommended Setup for Our Team

**Start simple:**

1. Use **Option A** (file-based, log to Google Drive) for Colab training
2. Use `mlflow.autolog()` with HuggingFace Trainer — zero code changes
3. View results locally with `mlflow ui`

**Later (when the team grows or we have more runs):**

1. Set up a shared MLflow server on a university VM or free-tier cloud instance
2. Use the Model Registry to track which models are deployed to the Jetson
3. Add ONNX export tracking for TensorRT deployment pipeline

---

## 9. Quick Reference

| Task | Command / Code |
|------|----------------|
| Install | `pip install mlflow` |
| Start server | `mlflow server --host 0.0.0.0 --port 5000` |
| Start UI only | `mlflow ui --port 5000` |
| Auto-log HuggingFace | `mlflow.autolog()` before `trainer.train()` |
| Log to Drive (Colab) | `mlflow.set_tracking_uri("file:///content/drive/MyDrive/mlflow-runs")` |
| View dashboard | `http://localhost:5000` |
| Compare runs | Select runs in UI → "Compare" |
