"""Command-line utility for training knowledge-distilled student models.

This script configures teacher/student architectures with :func:`load_model`,
builds FER dataloaders, and launches :func:`train_kd_model` using user-specified
hyperparameters parsed by :func:`parse_args`.
"""

import argparse
import torch
import torchvision.models as tv_models
from torch import nn

from fer import EmotionCNN, build_dataloaders, get_eval_transform, get_train_transform
from fer.distillation import train_kd_model


def load_model(arch: str, in_chans: int, num_classes: int, width_mult: float, ckpt: str | None, device: torch.device) -> nn.Module:
    if arch == "emotioncnn":
        model = EmotionCNN(in_chans=in_chans, num_classes=num_classes, width_mult=width_mult)
    elif arch == "resnet18":
        model = tv_models.resnet18(weights=None)
        if in_chans != 3:
            weight = model.conv1.weight
            model.conv1 = nn.Conv2d(in_chans, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
            if weight.shape[1] == 3 and in_chans == 1:
                model.conv1.weight.data = weight.mean(dim=1, keepdim=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if ckpt:
        try:
            state = torch.load(ckpt, map_location=device)
        except Exception as exc:  # pylint: disable=broad-except
            raise SystemExit(f"Failed to load checkpoint {ckpt}: {exc}") from exc

        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise SystemExit(
                f"Checkpoint {ckpt} did not contain a state_dict. "
                "Provide a file saved with torch.save(model.state_dict())."
            )

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise SystemExit(
                "Checkpoint parameters did not match the requested architecture. "
                "Verify --teacher-arch/--student-arch, in_chans, and width_mult "
                "align with the checkpoint that was trained.\n"
                f"Underlying error: {exc}"
            ) from exc

    model.to(device)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge distillation training entrypoint")
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv")
    parser.add_argument("--teacher-ckpt", required=True, help="Checkpoint for the teacher model")
    parser.add_argument("--ckpt-dir", default="runs/kd", help="Where to save student checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--student-in-chans", type=int, default=1, choices=[1, 3])
    parser.add_argument("--teacher-in-chans", type=int, default=3, choices=[1, 3])
    parser.add_argument("--student-width-mult", type=float, default=1.0)
    parser.add_argument("--teacher-width-mult", type=float, default=1.0)
    parser.add_argument("--student-arch", choices=["emotioncnn"], default="emotioncnn")
    parser.add_argument("--teacher-arch", choices=["emotioncnn", "resnet18"], default="emotioncnn")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--process-path", default=None)
    parser.add_argument("--device", default=None, help="Override device string")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = get_train_transform(args.student_in_chans)
    eval_tf = get_eval_transform(args.student_in_chans)
    train_loader, val_loader, _ = build_dataloaders(
        args.csv,
        batch_size=args.batch_size,
        num_workers=args.workers,
        in_chans=args.student_in_chans,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    teacher = load_model(
        args.teacher_arch,
        args.teacher_in_chans,
        args.num_classes,
        args.teacher_width_mult,
        args.teacher_ckpt,
        device,
    )
    student = load_model(
        args.student_arch,
        args.student_in_chans,
        args.num_classes,
        args.student_width_mult,
        ckpt=None,
        device=device,
    )

    stats = train_kd_model(
        student,
        teacher,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        T=args.temperature,
        alpha=args.alpha,
        ce_weight=args.ce_weight,
        ckpt_dir=args.ckpt_dir,
        history_path=args.history_path,
        process_path=args.process_path,
        args=args,
    )

    print("KD training complete. Best val acc:", stats["best_val_acc"])
    print("History saved to:", stats["history_path"])
    print("Process summary saved to:", stats["process_path"])


if __name__ == "__main__":
    main()
