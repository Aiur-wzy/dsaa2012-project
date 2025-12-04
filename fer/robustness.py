"""Noise and corruption robustness helpers for FER evaluation.

Functions here apply common image corruptions (blur, JPEG compression, noise),
while :func:`evaluate_with_corruptions` scores a model across all augmentations.
"""

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def add_brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def jpeg_compress(img: np.ndarray, quality: int = 50) -> np.ndarray:
    """Apply JPEG compression while preserving the original channel count."""

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # ``imencode`` returns a 3-channel image by default even for grayscale inputs.
    # Track the intended channel layout and decode accordingly to avoid silently
    # changing 1-channel batches into 3-channel images, which caused the model to
    # receive unexpected inputs during robustness evaluation.
    is_gray = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
    encode_img = img[..., 0] if img.ndim == 3 and img.shape[2] == 1 else img

    success, enc = cv2.imencode(".jpg", encode_img, encode_param)
    if not success:
        return img

    decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)
    if decoded is None:
        return img

    return decoded[..., None] if is_gray else decoded


def random_rotate(img: np.ndarray, max_angle: float = 30) -> np.ndarray:
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h))


def eval_under_corruption(model, loader, device: torch.device, corruption_fn) -> float:
    """Compute accuracy when every image in ``loader`` is corrupted on the fly."""

    def _to_uint8(batch: np.ndarray) -> np.ndarray:
        # Invert standardization (mean=0.5, std=0.5) if inputs are in [-1, 1]
        if batch.min() < 0.0 or batch.max() > 1.0:
            batch = 0.5 * batch + 0.5
        batch = np.clip(batch, 0.0, 1.0)
        return (batch * 255.0).astype("uint8")

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            expected_c = x.shape[1]
            # B, C, H, W -> B, H, W, C for OpenCV utilities
            x_np = x.permute(0, 2, 3, 1).cpu().numpy()
            x_uint8 = _to_uint8(x_np)

            corrupted = []
            for img in x_uint8:
                img_corr = corruption_fn(img)
                if img_corr.ndim == 2:  # Ensure channel dimension is preserved
                    img_corr = np.expand_dims(img_corr, axis=-1)

                # Align channel count with the original input (1-channel vs. 3-channel)
                if img_corr.shape[-1] != expected_c:
                    if expected_c == 1:
                        if img_corr.shape[-1] == 3:
                            img_corr = cv2.cvtColor(img_corr, cv2.COLOR_BGR2GRAY)[..., None]
                        else:
                            img_corr = img_corr[..., :1]
                    elif expected_c == 3:
                        if img_corr.shape[-1] == 1:
                            img_corr = cv2.cvtColor(img_corr, cv2.COLOR_GRAY2BGR)
                        else:
                            img_corr = img_corr[..., :3]
                img_corr = img_corr.astype("float32") / 255.0
                corrupted.append(img_corr)

            x_corr = torch.from_numpy(np.stack(corrupted, axis=0).transpose(0, 3, 1, 2)).to(device)
            y = y.to(device)

            logits = model(x_corr)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


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
