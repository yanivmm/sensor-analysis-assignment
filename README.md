# 🚗 Sensor Analysis and Stop Sign Behavior Assignment

This assignment consists of **two parts**, each designed to evaluate your ability to work with real-world sensor data and video footage collected from Nexar dashcams.

---

## Part 1 – IMU Signal Analysis

In this task, you'll analyze the behavior of a deployed machine learning model trained on IMU (Inertial Measurement Unit) accelerometer data. Nexar dashcams collect rich sensor data during vehicle operation, including acceleration along multiple axes, enabling the detection of events like collisions or sudden motion changes.

You're provided with pre-extracted features for the training and test sets (with labels), and raw signal files for the inference set. Your goal is to evaluate how the model performs on the inference data and diagnose any performance gaps relative to the original test set.

### 🎯 Goals

1. **Generate features** from the raw IMU inference signals.
2. **Load and run inference** using the provided model.
3. **Compare predictions** against the true labels in the manual annotation set.
4. **Evaluate and compare performance** across datasets.
5. **Perform EDA** to investigate potential causes of model degradation.
6. **Propose short-term and long-term solutions**.

### 📁 Repository Structure

```
├── data/
│   ├── train.csv                    # Pre-extracted features with labels
│   ├── test.csv                     # Pre-extracted features with labels
│   ├── inference.csv                # ❗ Generated features for inference (no labels)
│   ├── raw/
│   │   ├── train/
│   │   ├── test/
│   │   └── inference/               # Raw .npz files for inference
│
├── data/manual_annotation/
│   └── inference_labels.csv         # ✅ Ground truth for inference set
│
├── extract_features.py              # Feature extraction logic
├── imu_pipeline.py                  # Pipeline with trained model
├── visualization.py                 # Signal viewer for manual exploration
├── example.ipynb                    # Starter notebook
├── models/
│   └── imu_pipeline.pkl             # Pre-trained RandomForest model
└── requirements.txt
```

### 🛠 Setup

```bash
pip install -r requirements.txt
```

> Python 3.8+ is recommended.

---

## 🔍 Instructions

### Step 1 – Visualize IMU Signals (Optional but Recommended)

```python
from visualization import signal_viewer
from pathlib import Path

signal_viewer(
    data_dir=Path('data/raw/train'),
    labels_csv=Path('data/train.csv')
)
```

Use this to better understand the structure of the signals and class distribution.

### Step 2 – Generate Inference Features

```python
from extract_features import process_dataset
process_dataset('inference')
```

This will create `data/inference.csv` using the same logic as in train/test.
Note: Unlike `train.csv` and `test.csv`, this file does **not** contain labels.

### Step 3 – Load Labels for Inference Set

```python
import pandas as pd
labels = pd.read_csv("data/manual_annotation/inference_labels.csv")
```

### Step 4 – Run Model Predictions

```python
import joblib
from imu_pipeline import IMUPipeline

df_inf = pd.read_csv("data/inference.csv")
model = joblib.load("models/imu_pipeline.pkl")

preds = model.predict(df_inf)
probs = model.predict_proba(df_inf)
```

### Step 5 – Evaluate and Compare

- Evaluate performance on the inference set using standard metrics (accuracy, precision, recall, F1).
- Compare against performance on the test set (`test.csv`).
- Perform **EDA** (exploratory data analysis) on both test and inference sets:
- Identify, explain, and justify the reason for the performance discrepancy. Pinpoint the root cause

---

## Your Task Summary

1. Evaluate the model on the `inference.csv` predictions using `inference_labels.csv`.
2. Compare results to performance on `test.csv`.
3. Perform EDA to understand dataset differences.
4. Suggest an immediate workaround.
5. Propose a long-term fix.

---

## Part 2 – Stop Sign Behavior Annotation

In this task, you'll help define an annotation protocol for driver behavior near stop signs.

You're provided with dashcam video clips that were flagged as potentially involving stop sign interactions. Your job is to create a simple, effective labeling guide for remote annotators — maximizing label quality and training signal with minimal ambiguity.

### 📄 Task Details

Reference file:
```
Near Stop Sign Behavior - Annotation Instructions.pdf
```

Constraints:
- **Only video is available** (no metadata or sensor data).
- You may assume annotators are remote and have limited context.

### 📝 Deliverables

- A clear set of instructions for annotators.
- Assumptions or simplifications you made.
- A short explanation of how your labels support training an effective model.

Expected effort: ~1 hour.

### 🎥 Video Input

See:
```
videos/
```

---

## 📤 Final Submission

Please submit the following in your GitHub fork:

- `example.ipynb` with your full analysis for Part 1.
- A file `stop_sign_annotation_protocol.md` (or `.pdf`) for Part 2.

When ready, send the repository link to the recruiter.

---

Good luck! 🚦