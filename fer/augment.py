import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


def get_train_transform(in_chans: int = 1) -> A.Compose:
    """Albumentations pipeline mirroring the draft notebook."""
    if in_chans == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                value=0,
                p=0.7,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=4, min_width=4, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_baseline_train_transform(in_chans: int = 1) -> A.Compose:
    """Minimal augmentation for baseline ablations (normalize + tensor)."""
    if in_chans == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def get_eval_transform(in_chans: int = 1) -> A.Compose:
    if in_chans == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, torch.ones_like(y, dtype=torch.float32), torch.arange(len(y))
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
