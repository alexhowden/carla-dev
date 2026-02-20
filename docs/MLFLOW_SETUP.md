# MLflow Setup Guide

Track training experiments, compare model performance, and manage model artifacts. Replaces the manual `MODEL_REGISTRY.md` workflow.

---

## 1. Install & Start

```bash
pip install mlflow
```

### Register existing models

```bash
python scripts/register_models_mlflow.py
```

### Start the UI

```bash
mlflow ui --port 5000
```

Open **http://localhost:5000** to view the dashboard.

---

## 2. Integrate with Training

### HuggingFace Trainer (recommended — zero code changes)

```python
import mlflow
mlflow.autolog()
trainer.train()  # everything logged automatically
```

### Colab

Log to Google Drive (no server needed):

```python
mlflow.set_tracking_uri("file:///content/drive/MyDrive/mlflow-runs")
```

View locally after syncing: `mlflow ui --backend-store-uri ./mlflow-runs`

---

## 3. Quick Reference

| Task | Command / Code |
|------|----------------|
| Install | `pip install mlflow` |
| Register past models | `python scripts/register_models_mlflow.py` |
| Start UI | `mlflow ui --port 5000` |
| Auto-log HuggingFace | `mlflow.autolog()` before `trainer.train()` |
| Log to Drive (Colab) | `mlflow.set_tracking_uri("file:///content/drive/MyDrive/mlflow-runs")` |
| View dashboard | `http://localhost:5000` |
| Compare runs | Select runs in UI → "Compare" |
