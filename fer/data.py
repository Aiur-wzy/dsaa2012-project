import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

EMOTION_LABELS: Dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


class FER2013Dataset(Dataset):
    """Dataset wrapper for FER-2013.

    The original CSV ships with 48x48 grayscale images encoded as
    space-separated pixel strings. This wrapper handles usage filtering,
    reshaping, normalization, and optional channel replication when using
    backbones that expect three input channels.
    """

    def __init__(
        self,
        csv_or_df: os.PathLike | str | pd.DataFrame,
        usage: Optional[str] = None,
        transform: Optional[Callable] = None,
        in_chans: int = 1,
        return_group: bool = False,
    ) -> None:
        if isinstance(csv_or_df, pd.DataFrame):
            df = csv_or_df.copy()
        else:
            try:
                csv_path = os.fspath(csv_or_df)
            except TypeError:
                # Accept file-like objects or any pandas-compatible buffer
                try:
                    df = pd.read_csv(csv_or_df)
                except Exception as exc:  # pragma: no cover - defensive guard
                    raise TypeError(
                        "csv_or_df must be a path-like object, file-like object, or pandas DataFrame"
                    ) from exc
            else:
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(csv_path)
                df = pd.read_csv(csv_path)

        if usage is not None:
            df = df[df["Usage"] == usage].reset_index(drop=True)

        self.df = df
        self.transform = transform
        self.in_chans = in_chans
        self.return_group = return_group

        # Assign deterministic synthetic "age-like" groups for fairness analysis.
        rng = np.random.default_rng(42)
        self.synthetic_age_groups: List[str] = rng.choice(
            ["teen", "adult", "senior"], size=len(self.df)
        ).tolist()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        label = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32)
        if pixels.size != 48 * 48:
            raise ValueError(f"Unexpected pixel length {pixels.size} at index {idx}")
        image = pixels.reshape(48, 48)

        if self.in_chans == 3:
            image = np.stack([image, image, image], axis=-1)
        else:
            image = np.expand_dims(image, axis=-1)

        image = image / 255.0

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        else:
            tensor = image

        if self.return_group:
            group_label = self.synthetic_age_groups[idx]
            return tensor, label, group_label

        return tensor, label


def build_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    in_chans: int = 1,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders using FER2013 splits."""

    train_ds = FER2013Dataset(csv_path, usage="Training", transform=train_transform, in_chans=in_chans)
    val_ds = FER2013Dataset(csv_path, usage="PublicTest", transform=eval_transform, in_chans=in_chans)
    test_ds = FER2013Dataset(csv_path, usage="PrivateTest", transform=eval_transform, in_chans=in_chans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
