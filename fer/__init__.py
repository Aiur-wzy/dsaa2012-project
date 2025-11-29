"""A lightweight facial expression recognition toolkit.

The package bundles dataset utilities, augmentation helpers, a compact
convolutional network, training loops, and a simple OpenCV-based
inference pipeline for the FER-2013 dataset.
"""

from .data import FER2013Dataset, EMOTION_LABELS, build_dataloaders
from .augment import get_train_transform, get_eval_transform, mixup_data, mixup_criterion
from .models import EmotionCNN
from .train import train_model, evaluate, save_checkpoint, load_checkpoint
from .inference import FaceDetector, preprocess_face, run_realtime_demo, predict_image
from .robustness import add_brightness_contrast, add_gaussian_blur, jpeg_compress, random_rotate, topk_accuracy, group_metrics

__all__ = [
    "FER2013Dataset",
    "EMOTION_LABELS",
    "build_dataloaders",
    "get_train_transform",
    "get_eval_transform",
    "mixup_data",
    "mixup_criterion",
    "EmotionCNN",
    "train_model",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "FaceDetector",
    "preprocess_face",
    "run_realtime_demo",
    "predict_image",
    "add_brightness_contrast",
    "add_gaussian_blur",
    "jpeg_compress",
    "random_rotate",
    "topk_accuracy",
    "group_metrics",
]
