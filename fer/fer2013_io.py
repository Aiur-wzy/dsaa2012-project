"""Utilities for reading/writing FER-2013 style examples and running inference.

The helpers in this module make it easy to:
- Load a FER-2013 formatted CSV (with `emotion`, `pixels`, `Usage` columns)
  and run predictions with a trained model.
- Convert a standalone image (e.g., PNG) into a FER-2013 compatible example
  that can be appended to a CSV for quick experiments.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .augment import get_eval_transform
from .data import EMOTION_LABELS, FER2013Dataset


Prediction = Dict[str, float | int | str]


def predict_from_fer2013_csv(
    model: torch.nn.Module,
    csv_path: str,
    *,
    usage: Optional[str] = None,
    device: torch.device | str = "cpu",
    in_chans: int = 1,
    batch_size: int = 64,
    transform=None,
) -> List[Prediction]:
    """Run model predictions on a FER-2013 formatted CSV file.

    Args:
        model: Trained PyTorch model that outputs logits for seven emotions.
        csv_path: Path to the FER-2013 style CSV file.
        usage: Optional split name (`"Training"`, `"PublicTest"`, or
            `"PrivateTest"`) to filter rows before inference.
        device: Torch device string or object (e.g., ``"cuda"`` or ``"cpu"``).
        in_chans: Number of input channels expected by the model (1 or 3).
        batch_size: Batch size for inference.
        transform: Optional Albumentations transform; defaults to
            :func:`fer.augment.get_eval_transform`.

    Returns:
        A list of dictionaries containing ``label_index``, ``label_name``, and
        ``confidence`` for every row in the CSV.
    """

    if transform is None:
        transform = get_eval_transform(in_chans)

    dataset = FER2013Dataset(csv_path, usage=usage, transform=transform, in_chans=in_chans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    results: List[Prediction] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            scores, preds = probs.max(dim=1)
            for score, pred in zip(scores, preds):
                idx = int(pred.item())
                results.append(
                    {
                        "label_index": idx,
                        "label_name": EMOTION_LABELS[idx],
                        "confidence": float(score.item()),
                    }
                )

    return results


def image_to_fer2013_row(
    image_path: str,
    *,
    emotion: int = 0,
    usage: str = "Training",
    resize: bool = True,
) -> Dict[str, str | int]:
    """Convert an image file into a FER-2013 CSV row.

    The image is loaded in grayscale, optionally resized to 48×48, and flattened
    into a space-delimited pixel string matching the original FER-2013 format.

    Args:
        image_path: Path to an image file (PNG/JPG/etc.).
        emotion: Integer emotion label to store in the row (defaults to 0).
        usage: Usage split label to store (e.g., ``"Training"`` or ``"PrivateTest"``).
        resize: If True, resize the image to 48×48 before flattening.

    Returns:
        A dictionary with ``emotion``, ``pixels``, and ``Usage`` keys suitable for
        building a :class:`pandas.DataFrame` or appending to an existing CSV.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")

    if resize:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

    flat = image.reshape(-1)
    pixel_str = " ".join(str(int(v)) for v in flat)

    return {"emotion": int(emotion), "pixels": pixel_str, "Usage": usage}


def append_images_to_fer2013_csv(
    image_paths: Iterable[str],
    csv_path: str,
    *,
    emotion: int = 0,
    usage: str = "Training",
    resize: bool = True,
) -> pd.DataFrame:
    """Append one or more images to a FER-2013 CSV and return the updated DataFrame.

    This is a convenience wrapper around :func:`image_to_fer2013_row` for quickly
    constructing sample CSVs from image files.

    Args:
        image_paths: Collection of image paths to convert.
        csv_path: Destination CSV path to create or update.
        emotion: Integer emotion label to store for each converted image.
        usage: Usage split label to store for each converted image.
        resize: Whether to resize to 48×48 before flattening.

    Returns:
        The resulting :class:`pandas.DataFrame` containing the original and newly
        appended rows.
    """

    rows = [image_to_fer2013_row(path, emotion=emotion, usage=usage, resize=resize) for path in image_paths]
    new_df = pd.DataFrame(rows)

    try:
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        combined = new_df

    combined.to_csv(csv_path, index=False)
    return combined
