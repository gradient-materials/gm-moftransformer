# MOFTransformer version 2.2.0
"""Runtime compatibility helpers for PyTorch 1.x/2.x and Lightning 1.x/2.x."""

from __future__ import annotations

import inspect
from typing import Any, Union

import torch
import pytorch_lightning as pl


def lightning_major_version() -> int:
    """Return the major version of the installed pytorch-lightning package."""
    version = pl.__version__.split(".")[0]
    return int(version)


def is_lightning_v2() -> bool:
    """True when pytorch-lightning 2.x is installed."""
    return lightning_major_version() >= 2


def load_checkpoint(path: str, map_location: Union[str, torch.device] = "cpu") -> Any:
    """
    Load a Lightning checkpoint, compatible with torch 1.x and 2.x.

    PyTorch 2.x supports ``weights_only``; Lightning checkpoints need
    ``weights_only=False`` when that parameter exists.
    """
    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def resolve_accelerator(accelerator: str | None) -> str:
    """Prefer GPU when CUDA is available and accelerator is auto."""
    if accelerator in ("auto", None):
        if torch.cuda.is_available():
            return "gpu"
        return "cpu"
    return accelerator


def normalize_precision(precision: Union[int, str]) -> Union[int, str]:
    """Map legacy int precision flags to the active Lightning version."""
    if is_lightning_v2():
        if precision in (16, "16"):
            return "16-mixed"
        if precision in (32, "32"):
            return 32
        return precision
    if precision in ("16-mixed", "16"):
        return 16
    if precision in ("32", 32):
        return 32
    return precision


def get_trainer_strategy(is_interactive: bool) -> str | None:
    """DDP strategy for multi-device training, version-aware."""
    if is_interactive:
        return None
    if is_lightning_v2():
        return "ddp_find_unused_parameters_true"
    return "ddp"


def trainer_uses_benchmark() -> bool:
    """cudnn.benchmark is only passed to Trainer on Lightning 1.x."""
    return not is_lightning_v2()
