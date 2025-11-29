import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def add_brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def jpeg_compress(img: np.ndarray, quality: int = 50) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, 1)


def random_rotate(img: np.ndarray, max_angle: float = 30) -> np.ndarray:
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h))


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 2) -> float:
    topk = torch.topk(logits, k, dim=1).indices.cpu().numpy()
    labels_np = labels.cpu().numpy()
    correct = sum(1 for i in range(len(labels_np)) if labels_np[i] in topk[i])
    return correct / len(labels_np)


def group_metrics(y_true, y_pred, groups):
    metrics = {}
    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro")
        metrics[g] = {"acc": acc, "f1": f1}
    return metrics
