from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .augment import get_eval_transform
from .data import EMOTION_LABELS


@dataclass
class FaceDetector:
    """Wrapper around OpenCV's Haar cascade or DNN face detector."""

    detector_type: str = "haar"
    score_threshold: float = 0.7

    def __post_init__(self):
        if self.detector_type == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.model = cv2.CascadeClassifier(cascade_path)
        elif self.detector_type == "dnn":
            proto = cv2.data.haarcascades + "deploy.prototxt"
            weights = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            self.model = cv2.dnn.readNetFromCaffe(proto, weights)
        else:
            raise ValueError("detector_type must be 'haar' or 'dnn'")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector_type == "haar":
            faces = self.model.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()[0, 0]
        boxes: List[Tuple[int, int, int, int]] = []
        for det in detections:
            confidence = float(det[2])
            if confidence < self.score_threshold:
                continue
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


def preprocess_face(face_bgr: np.ndarray, in_chans: int = 1) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype("float32") / 255.0
    if in_chans == 3:
        face = np.stack([face, face, face], axis=-1)
    else:
        face = np.expand_dims(face, axis=-1)
    return face


def predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transform=None,
    device: torch.device | str = "cpu",
    in_chans: int = 1,
) -> Tuple[str, float]:
    if transform is None:
        transform = get_eval_transform(in_chans)

    processed = transform(image=image)["image"].unsqueeze(0).to(device)
    logits = model(processed)
    prob = torch.softmax(logits, dim=1)
    score, pred = prob.max(dim=1)
    label = EMOTION_LABELS[int(pred.item())]
    return label, float(score.item())


def run_realtime_demo(
    model: torch.nn.Module,
    detector: FaceDetector,
    device: torch.device | str = "cpu",
    in_chans: int = 1,
) -> None:
    transform = get_eval_transform(in_chans)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    model.to(device)
    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        for (x, y, w, h) in boxes:
            face = frame[y : y + h, x : x + w]
            pre = preprocess_face(face, in_chans=in_chans)
            tensor = transform(image=pre)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)
                score, pred = prob.max(dim=1)
                label = EMOTION_LABELS[int(pred.item())]
            text = f"{label} {score.item():.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("FER", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
