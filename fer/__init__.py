"""A lightweight facial expression recognition toolkit.

The package bundles dataset utilities, augmentation helpers, a compact
convolutional network, training loops, and a simple OpenCV-based
inference pipeline for the FER-2013 dataset.
"""

from .data import FER2013Dataset, EMOTION_LABELS, build_dataloaders
from .analysis import evaluate_with_confusion, save_confusion_outputs
from .fairness_analysis import compute_age_groups, compute_confidence_groups, plot_group_metrics
from .augment import get_train_transform, get_eval_transform, get_baseline_train_transform, mixup_data, mixup_criterion
from .losses import LabelSmoothingCE
from .models import EmotionCNN, count_parameters
from .distillation import distillation_loss, train_kd_model
from .pruning import structured_prune_model, prune_and_finetune
from .quantization import quantize_dynamic, prepare_qat_model, convert_qat_model
from .train import train_model, evaluate, save_checkpoint, load_checkpoint
from .inference import FaceDetector, preprocess_face, run_realtime_demo, predict_image
from .fer2013_io import predict_from_fer2013_csv, image_to_fer2013_row, append_images_to_fer2013_csv
from .robustness import (
    add_brightness_contrast,
    add_gaussian_blur,
    eval_under_corruption,
    jpeg_compress,
    random_rotate,
    topk_accuracy,
    group_metrics,
)

__all__ = [
    "FER2013Dataset",
    "EMOTION_LABELS",
    "build_dataloaders",
    "get_train_transform",
    "get_eval_transform",
    "get_baseline_train_transform",
    "mixup_data",
    "mixup_criterion",
    "LabelSmoothingCE",
    "EmotionCNN",
    "count_parameters",
    "distillation_loss",
    "train_kd_model",
    "structured_prune_model",
    "prune_and_finetune",
    "quantize_dynamic",
    "prepare_qat_model",
    "convert_qat_model",
    "train_model",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "evaluate_with_confusion",
    "save_confusion_outputs",
    "compute_confidence_groups",
    "compute_age_groups",
    "plot_group_metrics",
    "FaceDetector",
    "preprocess_face",
    "run_realtime_demo",
    "predict_image",
    "predict_from_fer2013_csv",
    "image_to_fer2013_row",
    "append_images_to_fer2013_csv",
    "add_brightness_contrast",
    "add_gaussian_blur",
    "eval_under_corruption",
    "jpeg_compress",
    "random_rotate",
    "topk_accuracy",
    "group_metrics",
]
