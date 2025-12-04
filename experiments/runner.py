"""Unified experiment runner for FER ablations.

The script accepts a YAML/JSON config describing multiple experiment
variants, launches each training run programmatically, evaluates the
resulting model on the test split, and writes a consolidated summary CSV.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from fer import (
    EmotionCNN,
    build_dataloaders,
    evaluate,
    get_baseline_train_transform,
    get_eval_transform,
    get_train_transform,
    load_checkpoint,
    save_process_file,
    train_model,
)


DEFAULTS: Dict[str, Any] = {
    "csv": "fer2013.csv",
    "batch_size": 128,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "workers": 4,
    "in_chans": 1,
    "width_mult": 1.0,
    "mixup_alpha": 0.2,
    "label_smoothing_eps": 0.1,
    "num_classes": 7,
    "augmentation": "full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of FER experiments from a config file")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML or JSON experiment config")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("PyYAML is required to parse YAML configs") from exc
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_train_transform_from_name(augmentation: str, in_chans: int):
    if augmentation == "baseline":
        return get_baseline_train_transform(in_chans)
    return get_train_transform(in_chans)


def prepare_dataloaders(
    csv_path: Path,
    batch_size: int,
    workers: int,
    in_chans: int,
    augmentation: str,
):
    train_tf = get_train_transform_from_name(augmentation, in_chans)
    eval_tf = get_eval_transform(in_chans)
    return build_dataloaders(
        str(csv_path),
        batch_size=batch_size,
        num_workers=workers,
        in_chans=in_chans,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )


def resolve_path(base_dir: Path, maybe_path: str | Path) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def run_experiment(
    exp_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    config_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if "name" not in exp_cfg:
        raise ValueError("Each experiment must define a 'name' field")
    name = exp_cfg["name"]

    resolved_cfg = deepcopy(base_cfg)
    resolved_cfg.update(exp_cfg.get("train_args", {}))

    ckpt_dir = resolve_path(config_dir, exp_cfg.get("ckpt_dir", f"runs/{name}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_path(config_dir, resolved_cfg["csv"])
    batch_size = int(resolved_cfg.get("batch_size", DEFAULTS["batch_size"]))
    workers = int(resolved_cfg.get("workers", DEFAULTS["workers"]))
    in_chans = int(resolved_cfg.get("in_chans", DEFAULTS["in_chans"]))
    augmentation = resolved_cfg.get("augmentation", DEFAULTS["augmentation"])
    width_mult = float(resolved_cfg.get("width_mult", DEFAULTS["width_mult"]))

    train_loader, val_loader, test_loader = prepare_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        workers=workers,
        in_chans=in_chans,
        augmentation=augmentation,
    )

    model = EmotionCNN(in_chans=in_chans, width_mult=width_mult).to(device)

    stats = train_model(
        model,
        train_loader,
        val_loader,
        epochs=int(resolved_cfg.get("epochs", DEFAULTS["epochs"])),
        device=device,
        lr=float(resolved_cfg.get("lr", DEFAULTS["lr"])),
        weight_decay=float(resolved_cfg.get("weight_decay", DEFAULTS["weight_decay"])),
        use_mixup=bool(resolved_cfg.get("mixup", False)),
        mixup_alpha=float(resolved_cfg.get("mixup_alpha", DEFAULTS["mixup_alpha"])),
        ckpt_dir=ckpt_dir,
        loss=resolved_cfg.get("loss", "ce"),
        label_smoothing_eps=float(
            resolved_cfg.get("label_smoothing_eps", DEFAULTS["label_smoothing_eps"])
        ),
        num_classes=int(resolved_cfg.get("num_classes", DEFAULTS["num_classes"])),
    )

    # Persist process summary
    process_payload = {
        "name": name,
        "csv": str(csv_path),
        "ckpt_dir": str(ckpt_dir),
        "augmentation": augmentation,
        "width_mult": width_mult,
        **resolved_cfg,
    }
    save_process_file(ckpt_dir, process_payload, stats)

    # Evaluate on the test split using the best checkpoint
    best_ckpt = ckpt_dir / "best.pt"
    load_checkpoint(model, best_ckpt, device)
    criterion = torch.nn.CrossEntropyLoss()
    _test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "name": name,
        "loss": resolved_cfg.get("loss", "ce"),
        "augmentation": augmentation,
        "mixup": bool(resolved_cfg.get("mixup", False)),
        "width_mult": width_mult,
        "val_acc": stats.get("best_val_acc", 0.0),
        "test_acc": test_acc,
        "ckpt_path": str(best_ckpt),
    }


def run_all_experiments(config: Dict[str, Any], config_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    experiments = config.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Config file must contain an 'experiments' list")

    base_cfg = deepcopy(DEFAULTS)
    base_cfg.update({k: v for k, v in config.items() if k != "experiments"})

    summary_rows = []
    for exp_cfg in experiments:
        summary_rows.append(run_experiment(exp_cfg, base_cfg, config_dir, device))

    return summary_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dir = args.config.parent.resolve()

    summary_rows = run_all_experiments(config, config_dir, device)

    summary_path = resolve_path(config_dir, config.get("summary_path", "experiments_summary.csv"))
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary for {len(summary_rows)} experiments to {summary_path}")


if __name__ == "__main__":
    main()
