from typing import Iterable, Type

import torch
from torch import nn
from torch.ao.quantization import convert, get_default_qat_qconfig, prepare_qat


def quantize_dynamic(model: nn.Module, modules: Iterable[Type[nn.Module]] | None = None, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Wrap ``torch.quantization.quantize_dynamic`` with sensible defaults."""

    modules = set(modules) if modules is not None else {nn.Linear}
    return torch.quantization.quantize_dynamic(model, modules, dtype=dtype)


def prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Attach fake-quantization modules for QAT training."""

    model.qconfig = get_default_qat_qconfig(backend)
    model.train()
    return prepare_qat(model)


def convert_qat_model(model: nn.Module) -> nn.Module:
    """Convert a QAT-trained model to an INT8 inference graph."""

    model.eval()
    return convert(model)
