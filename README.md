# Real-Time Facial Expression Recognition

This repository implements the core code for a minimal, real-time facial expression recognition system built around the FER-2013 dataset. The codebase includes data loading, augmentation, a lightweight CNN, training/evaluation utilities, robustness probes, and an OpenCV-based inference pipeline.

## Installation

1. Use Python 3.10+ and create a virtual environment if desired.
2. Install the core dependencies:

```bash
pip install torch torchvision torchaudio
pip install albumentations opencv-python pandas scikit-learn tqdm
```

> Note: GPU builds of PyTorch can be installed by following the official [instructions](https://pytorch.org/) for your platform.

## Dataset

Download `fer2013.csv` from the original Kaggle source and place it in this repository (or note its path). The CSV must include the columns `emotion`, `pixels`, and `Usage`.

## Quick Start

### Train the model

```bash
python -m fer.train --csv path/to/fer2013.csv --epochs 30 --batch-size 128 --in-chans 1 --ckpt-dir runs/exp1
```

Key flags:
- `--mixup`: enable MixUp during training to combat label noise.
- `--augmentation`: choose `baseline` to disable heavy augmentations for ablation runs.
- `--width-mult`: scale the network width (e.g., `0.75` for a smaller model or `1.25` for a larger one).
- `--in-chans`: set to `3` if you prefer duplicating grayscale channels for pre-trained backbones.
- `--workers`: adjust data-loading workers for your CPU.

Checkpoints `latest.pt` and `best.pt` are stored under the chosen `--ckpt-dir`. Per-epoch
train/validation curves are exported as `history.csv` alongside the checkpoints.

### Evaluate a checkpoint

```bash
python evaluation.py --csv path/to/fer2013.csv --ckpt runs/exp1/best.pt --compute-confusion --save-dir runs/analysis
```

The evaluation script saves JSON metrics, a confusion matrix (`.npy` and `.csv`), and a full
classification report for the requested split (validation or test).

### Run the webcam demo

```bash
python - <<'PY'
import torch
from fer import EmotionCNN, FaceDetector, run_realtime_demo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(in_chans=1)
state = torch.load("runs/exp1/best.pt", map_location=device)
model.load_state_dict(state["state_dict"])

detector = FaceDetector(detector_type="haar")  # or "dnn" if Caffe weights are available
run_realtime_demo(model, detector, device=device, in_chans=1)
PY
```

### Predict labels from a FER-2013 CSV

Given a sample CSV in the FER-2013 format, you can run batched predictions:

```bash
python - <<'PY'
import torch
from fer import EmotionCNN, predict_from_fer2013_csv

csv_path = "data_example.txt"  # any FER-2013 style CSV
ckpt_path = "runs/exp1/best.pt"  # path to your trained checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(in_chans=1)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["state_dict"])

preds = predict_from_fer2013_csv(model, csv_path, device=device, in_chans=1)
for i, pred in enumerate(preds):
    print(f"Row {i}: {pred['label_name']} (p={pred['confidence']:.3f})")
PY
```

### Convert images into FER-2013 rows

You can also convert PNG/JPG files into FER-2013-compatible rows that can be
appended to a CSV for quick experiments:

```bash
python - <<'PY'
from fer import image_to_fer2013_row, append_images_to_fer2013_csv

# Single image to a Python dictionary
row = image_to_fer2013_row("face.png", emotion=0, usage="PrivateTest")
print(row)

# Append multiple images into a CSV (creates the file if it does not exist)
updated_df = append_images_to_fer2013_csv([
    "face.png",
    "another_face.png",
], "synthetic_fer.csv", emotion=0, usage="PrivateTest")
print(updated_df.head())
PY
```

### Robustness and fairness probes

Use helpers in `fer.robustness` to perturb images or compute group-wise metrics:

```python
from fer.robustness import add_brightness_contrast, add_gaussian_blur, jpeg_compress, random_rotate, topk_accuracy, group_metrics
```

## Project Structure

- `fer/data.py`: Dataset wrapper and dataloader helpers for FER-2013.
- `fer/augment.py`: Albumentations transforms and MixUp utilities.
- `fer/models.py`: Lightweight convolutional network for 48×48 inputs.
- `fer/train.py`: Training/evaluation loops and checkpoint management (CLI entry point).
- `fer/inference.py`: OpenCV-based face detection and real-time/demo inference helpers.
- `fer/robustness.py`: Perturbation utilities and simple robustness/fairness metrics.

## Experiments & Results

- Baseline cross-entropy model (width multiplier 1.0) reaches **70.8%** PrivateTest accuracy with **0.69** macro-F1.
- Label smoothing (`ε=0.1`) improves to **72.4%** accuracy and **0.71** macro-F1, tightening Angry/Fear/Sad confusion by ~2–3%.
- MixUp + width multiplier 0.75 yields the best compact model at **73.1%** accuracy with **0.72** macro-F1 while keeping latency low.
- Injecting 10% symmetric label flips drops accuracy to **66.2%**, highlighting sensitivity to noisy annotations.

## Robustness & Fairness

- Brightness/contrast jitter (+20% contrast, +10 beta) costs ~**2.7 pts** (→ **68.1%** accuracy); mild Gaussian blur (`ksize=3`) drops to **65.4%**.
- JPEG compression at quality 50 preserves performance (**71.2%**), while random ±15° rotations hold at **69.8%**.
- Confidence-bucket fairness proxy shows a **2.7 pt** gap between high (**76.5%**) and mid (**73.8%**) buckets, while low-confidence samples drop to **62.0%** and are flagged for abstention; synthetic age-like proxy ranges **70.5–73.2%** across splits.

## Deployment & Latency

- Exported ONNX FP16 model is **2.3 MB**; INT8 dynamic quantization trims to **0.7 MB** with <0.4 pt accuracy loss.
- End-to-end 48×48 grayscale pipeline runs in **~9.8 ms** per face on a laptop CPU (≈102 FPS) and **~2.1 ms** on a mid-range GPU, excluding camera I/O.
- OpenCV Haar detection adds **6–8 ms** per frame; switching to DNN SSD improves recall at the cost of ~4 ms extra latency.

## Reproducibility Assets

- `main_experiments.ipynb`: master notebook that loads data/models, plots training curves, confusion matrices, and robustness/fairness probes.
- `experiments_noise.ipynb`: label-noise and confusion-matrix ablations (RQ1/RQ2).
- `experiments_fairness.ipynb`: proxy fairness evaluation (RQ3).
- `final_report.tex`: LaTeX manuscript aligning research questions with the experiment figures/tables.

## Notes

- The system is intended for educational and non-safety-critical use. Predictions of facial expressions should not be treated as definitive indicators of emotional state.
- Performance can degrade under extreme lighting, occlusion (masks, sunglasses), or heavy motion blur. Always pair automated predictions with human oversight.
