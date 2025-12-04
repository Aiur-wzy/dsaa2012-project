# Real-Time Facial Expression Recognition Toolkit

A lightweight, end-to-end pipeline for the FER-2013 dataset covering data prep, training, robustness/fairness analysis, compression, deployment exports, and live demos. Everything runs from Python entry points so you can reproduce experiments or plug the model into your own projects.

## 1) Environment Setup

1. Use Python 3.10+ (set up a virtual environment if you like).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

PyTorch CUDA wheels are not pinned; install the CUDA-enabled build that matches your GPU before running `pip install -r requirements.txt`.

## 2) Prepare the FER-2013 Dataset

1. Download `fer2013.csv` from Kaggle.
2. Place it at the project root (or note an absolute path). Required columns: `emotion`, `pixels`, and `Usage` with values `Training`, `PublicTest`, and `PrivateTest`.
3. For quick sanity checks, `data_example.txt` is a tiny FER-2013-style sample.

## 3) Train a Baseline Model

Launch the bundled trainer (dataloaders, augmentations, checkpoints, process metadata):

```bash
python -m fer.train --csv fer2013.csv --epochs 30 --batch-size 128 --in-chans 1 --ckpt-dir runs/exp1
```

Key flags:
- `--augmentation {full,baseline}` toggles heavy vs minimal augmentation.
- `--mixup`/`--mixup-alpha` enable MixUp.
- `--loss {ce,label_smoothing}` plus `--label-smoothing-eps` choose the loss.
- `--width-mult` scales the CNN; `--in-chans 3` replicates grayscale to RGB.
- `--workers` controls dataloader workers; set to 0 on CPU-only/Windows.

Artifacts in `--ckpt-dir`: `latest.pt`, `best.pt`, per-epoch `history.csv`, and `process.json` (consumed by notebooks).【F:fer/train.py†L111-L193】【F:fer/train.py†L201-L273】

## 4) Evaluate a Checkpoint

Compute metrics on validation (`PublicTest`) or test (`PrivateTest`) splits:

```bash
python evaluation.py --csv fer2013.csv --ckpt runs/exp1/best.pt --split test --compute-confusion --save-dir runs/analysis
```

Outputs: JSON metrics, optional confusion-matrix `.npy/.csv`, and a classification report. The script rebuilds the model with your `--in-chans`/`--width-mult` before scoring.【F:evaluation.py†L1-L108】

## 5) Robustness Probes

Sweep common corruptions (brightness/contrast, blur, JPEG, rotations):

```bash
python robust_eval.py --csv fer2013.csv --ckpt runs/exp1/best.pt --split test --output-dir runs/robustness
```

The script reports clean accuracy, per-corruption metrics, a CSV table, Matplotlib plot, and markdown summary.【F:robust_eval.py†L1-L174】

## 6) Fairness-Oriented Evaluation

Assess confidence-based and synthetic age-group gaps:

```bash
python fairness_cli.py --csv fer2013.csv --ckpt runs/exp1/best.pt --usage PrivateTest --output-dir runs/fairness
```

The CLI saves groupwise accuracy/F1 CSVs, renders accuracy plots, and optionally emits a markdown summary highlighting the largest gaps.【F:fairness_cli.py†L1-L181】

## 7) Knowledge Distillation Training

Train a compact student against a larger teacher:

```bash
python kd_train.py --csv fer2013.csv --teacher-ckpt runs/teacher/best.pt --teacher-arch resnet18 --teacher-in-chans 3 --student-arch emotioncnn --student-in-chans 1 --ckpt-dir runs/kd --epochs 30
```

You can pick `emotioncnn` or `resnet18` teachers, set temperature/alpha, and log history/process summaries for downstream analysis.【F:kd_train.py†L1-L116】

## 8) Structured Pruning + Fine-Tuning

Prune channels then recover accuracy with a short fine-tune:

```bash
python prune_cli.py --csv fer2013.csv --ckpt runs/exp1/best.pt --prune-amount 0.3 --finetune-epochs 5 --output-dir runs/pruned
```

Outputs include pruned checkpoints, best/final validation accuracy, and size estimates for the compressed model.【F:prune_cli.py†L1-L80】

## 9) Quantization (Dynamic or QAT)

Dynamic quantization for quick INT8 exports:

```bash
python qat_cli.py --mode dynamic --csv fer2013.csv --ckpt runs/exp1/best.pt
```

Quantization-aware training fine-tune:

```bash
python qat_cli.py --mode qat --csv fer2013.csv --ckpt runs/exp1/best.pt --epochs 5 --batch-size 128
```

The CLI prepares dataloaders, runs QAT epochs, converts to INT8, and saves the quantized state_dict; dynamic mode performs one-shot quantization on CPU.【F:qat_cli.py†L1-L104】【F:qat_cli.py†L106-L168】

## 10) Export for Deployment

Export TorchScript/ONNX artifacts and benchmark latency (optional quantization):

```bash
python export_model.py --checkpoint runs/exp1/best.pt --output-dir exports --in-chans 1 --quantize --evaluate-csv fer2013.csv
```

Generates TorchScript, ONNX with dynamic batch axes, latency measurements, and (if `--quantize`) dynamically quantized TorchScript plus optional accuracy estimates.【F:export_model.py†L1-L126】

## 11) Inference Utilities

Batch predictions from a FER-2013 CSV:

```bash
python predict_from_csv_cli.py --ckpt runs/exp1/best.pt --csv data_example.txt --usage all --in-chans 1
```

Convert images into FER-2013 rows (create/append to CSV):

```bash
python images_to_fer_cli.py --image face.png --images face.png another_face.png --output synthetic_fer.csv --emotion 0 --usage PrivateTest
```

Both utilities normalize/reshape inputs, preview outputs, and respect FER header/usage conventions.【F:predict_from_csv_cli.py†L1-L35】【F:fer/fer2013_io.py†L23-L148】

## 12) Live Demo & Detector Benchmarking

Webcam demo with on-screen labels and FPS diagnostics:

```bash
python webcam_demo_cli.py --ckpt runs/exp1/best.pt --in-chans 1 --detector haar
```

Headless detector/inference timing benchmark (webcam or video file):

```bash
python detector_benchmark.py --ckpt runs/exp1/best.pt --output runs/bench/metrics.json --frames 500 --detector dnn
```

Both scripts reuse `FaceDetector`, alignment helpers, and preprocessing to 48×48 tensors before running the model.【F:webcam_demo_cli.py†L1-L35】【F:detector_benchmark.py†L1-L113】

## 13) Notebooks & Autogenerated Scripts

- `main_experiments.ipynb` / `main_experiments.py`: core training curves, confusion matrices, robustness/fairness summaries (consumes `history.csv`/`process.json`).【F:main_experiments.py†L1-L113】
- `experiments_noise.ipynb` / `experiments_noise.py`: label-noise ablations and confusion matrices.【F:experiments_noise.py†L1-L119】
- `experiments_fairness.ipynb`: explores synthetic age-group metrics built into `FER2013Dataset` (set `return_group=True`).【F:fer/data.py†L1-L62】

Run notebooks with Jupyter or execute the exported `.py` files after setting `CSV_PATH` and other constants near the top.

## 14) Repository Map

- `fer/`: datasets, augmentations, models, training loop, inference helpers, robustness utilities, fairness analysis, quantization, and pruning.【F:fer/__init__.py†L1-L38】【F:fer/data.py†L1-L62】【F:fer/quantization.py†L1-L105】【F:fer/pruning.py†L1-L100】
- Top-level scripts: training/eval (`fer/train.py`, `evaluation.py`), robustness (`robust_eval.py`), fairness (`fairness_cli.py`), distillation (`kd_train.py`), compression (`prune_cli.py`, `qat_cli.py`), deployment (`export_model.py`), demos/benchmarks (`webcam_demo_cli.py`, `detector_benchmark.py`), and notebook exports for experiments.【F:evaluation.py†L1-L108】【F:robust_eval.py†L1-L174】【F:kd_train.py†L1-L116】【F:qat_cli.py†L1-L168】【F:prune_cli.py†L1-L80】【F:detector_benchmark.py†L1-L113】

## 15) Tips for Smooth Runs

- Match `--in-chans` and `--width-mult` between training and downstream scripts (evaluation, export, demo, KD, pruning, QAT).
- Use `--workers 0` on CPU-only/Windows to avoid dataloader multiprocessing issues.
- Press `q` to exit the webcam demo; pass `--video` to `detector_benchmark.py` for offline runs.
- Validate `pixels` strings are length 2304 (48×48) when crafting new CSV rows or image conversions to avoid dataset errors.
