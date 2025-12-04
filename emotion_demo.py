"""Minimal entry point for running the real-time emotion recognition demo.

This module loads a trained :class:`EmotionCNN`, initializes a chosen
:class:`FaceDetector`, and starts the webcam-based demo via
:func:`fer.run_realtime_demo`.
"""

import torch
from fer import EmotionCNN, FaceDetector, run_realtime_demo

# 设备选择（CUDA优先，否则CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
model = EmotionCNN(in_chans=1)
state = torch.load("runs/exp1/best.pt", map_location=device)
model.load_state_dict(state["state_dict"])
# 初始化人脸检测器（haar或dnn）
detector = FaceDetector(detector_type="haar")  # 若有Caffe权重可换"dnn"
# 运行实时演示
run_realtime_demo(model, detector, device=device, in_chans=1)