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
- `--in-chans`: set to `3` if you prefer duplicating grayscale channels for pre-trained backbones.
- `--workers`: adjust data-loading workers for your CPU.

Checkpoints `latest.pt` and `best.pt` are stored under the chosen `--ckpt-dir`.

### Evaluate a checkpoint

```bash
python - <<'PY'
import torch
from fer import EmotionCNN, evaluate, build_dataloaders, get_eval_transform

csv_path = "path/to/fer2013.csv"
ckpt_path = "runs/exp1/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN(in_chans=1).to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["state_dict"])

_, _, test_loader = build_dataloaders(csv_path, batch_size=256, num_workers=4, in_chans=1, eval_transform=get_eval_transform(1))
criterion = torch.nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print({"test_loss": test_loss, "test_acc": test_acc})
PY
```

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

## Notes

- The system is intended for educational and non-safety-critical use. Predictions of facial expressions should not be treated as definitive indicators of emotional state.
- Performance can degrade under extreme lighting, occlusion (masks, sunglasses), or heavy motion blur. Always pair automated predictions with human oversight.
