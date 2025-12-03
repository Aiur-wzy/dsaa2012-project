from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import time

from .augment import get_eval_transform
from .data import EMOTION_LABELS


_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def align_face(face_bgr: np.ndarray) -> np.ndarray:
    """Rotate the cropped face so that the eyes are roughly horizontal.

    This uses OpenCV's Haar eye detector for a lightweight heuristic. If two
    eyes are not found the original face is returned unchanged.
    """

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = _EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if len(eyes) < 2:
        return face_bgr

    # Pick the two most confident (largest) detections and sort by x position.
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])

    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    left_eye_center = (x1 + w1 / 2.0, y1 + h1 / 2.0)
    right_eye_center = (x2 + w2 / 2.0, y2 + h2 / 2.0)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (face_bgr.shape[1] / 2.0, face_bgr.shape[0] / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        face_bgr,
        rot_mat,
        (face_bgr.shape[1], face_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


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

    total_detection = 0.0
    total_preprocess = 0.0
    total_inference = 0.0
    frame_count = 0
    loop_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_start = time.perf_counter()
        boxes = detector.detect(frame)
        total_detection += time.perf_counter() - detect_start

        for (x, y, w, h) in boxes:
            face = frame[y : y + h, x : x + w]
            prep_start = time.perf_counter()
            aligned = align_face(face)
            pre = preprocess_face(aligned, in_chans=in_chans)
            tensor = transform(image=pre)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                total_preprocess += time.perf_counter() - prep_start
                infer_start = time.perf_counter()
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)
                score, pred = prob.max(dim=1)
                total_inference += time.perf_counter() - infer_start
                label = EMOTION_LABELS[int(pred.item())]
            text = f"{label} {score.item():.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_count += 1
        cv2.imshow("FER", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    loop_time = time.perf_counter() - loop_start
    if frame_count > 0:
        fps = frame_count / loop_time
        print(
            f"Processed {frame_count} frames | "
            f"avg FPS: {fps:.2f} | detection: {total_detection / frame_count * 1000:.1f} ms | "
            f"preprocess: {total_preprocess / frame_count * 1000:.1f} ms | "
            f"inference: {total_inference / frame_count * 1000:.1f} ms"
        )

    cap.release()
    cv2.destroyAllWindows()
