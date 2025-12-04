"""Benchmark detector variants in a headless loop and summarize timings."""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import torch

from fer import EmotionCNN, FaceDetector, get_eval_transform, preprocess_face
from fer.inference import align_face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a headless detector/inference benchmark over webcam or video frames."
    )
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--in-chans", type=int, default=1, choices=[1, 3], help="Input channels for the model.")
    parser.add_argument("--detector", choices=["haar", "dnn"], default="haar", help="Detector backend to benchmark.")
    parser.add_argument(
        "--alignment",
        choices=["on", "off"],
        default="on",
        help="Toggle eye-based alignment before preprocessing.",
    )
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to process before exiting.")
    parser.add_argument("--output", required=True, help="Destination JSON file for benchmark metrics.")
    parser.add_argument("--video", help="Optional video file to benchmark instead of the webcam.")
    parser.add_argument("--device", default=None, help="Override device string (cuda or cpu).")
    return parser.parse_args()


def build_model(ckpt_path: Path, in_chans: int, device: torch.device) -> EmotionCNN:
    model = EmotionCNN(in_chans=in_chans)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_benchmark(
    model: torch.nn.Module,
    detector: FaceDetector,
    frames: int,
    device: torch.device,
    in_chans: int,
    align_faces: bool,
    video_path: Optional[Path] = None,
) -> dict:
    transform = get_eval_transform(in_chans)
    source = str(video_path) if video_path else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        source_label = f"video source: {video_path}" if video_path else "webcam"
        raise RuntimeError(f"Cannot open {source_label}")

    total_detection = 0.0
    total_preprocess = 0.0
    total_inference = 0.0
    frame_count = 0
    loop_start = time.perf_counter()

    while frame_count < frames:
        ret, frame = cap.read()
        if not ret:
            break

        detect_start = time.perf_counter()
        boxes = detector.detect(frame)
        total_detection += time.perf_counter() - detect_start

        for (x, y, w, h) in boxes:
            face = frame[y : y + h, x : x + w]
            prep_start = time.perf_counter()
            aligned = align_face(face) if align_faces else face
            pre = preprocess_face(aligned, in_chans=in_chans)
            tensor = transform(image=pre)["image"].unsqueeze(0).to(device)
            total_preprocess += time.perf_counter() - prep_start

            with torch.no_grad():
                infer_start = time.perf_counter()
                _ = model(tensor)
                total_inference += time.perf_counter() - infer_start

        frame_count += 1

    loop_time = time.perf_counter() - loop_start
    cap.release()

    if frame_count == 0:
        raise RuntimeError("No frames processed; check the video source and frame count.")

    fps = frame_count / loop_time if loop_time > 0 else 0.0
    return {
        "frames": frame_count,
        "fps": fps,
        "det_ms": (total_detection / frame_count) * 1000.0,
        "prep_ms": (total_preprocess / frame_count) * 1000.0,
        "infer_ms": (total_inference / frame_count) * 1000.0,
    }


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    video_path = Path(args.video) if args.video else None
    if video_path and not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = FaceDetector(detector_type=args.detector)
    model = build_model(ckpt_path, args.in_chans, device)

    metrics = run_benchmark(
        model,
        detector,
        frames=args.frames,
        device=device,
        in_chans=args.in_chans,
        align_faces=args.alignment == "on",
        video_path=video_path,
    )

    metrics.update({"detector": args.detector, "alignment": args.alignment == "on"})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Benchmark complete. Results written to {output_path}")


if __name__ == "__main__":
    main()
