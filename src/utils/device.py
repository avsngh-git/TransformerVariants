"""Device detection and GPU information utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DeviceInfo:
    """Information about the selected compute device."""

    device: torch.device
    device_name: str
    gpu_memory_gb: float | None
    supports_bf16: bool
    supports_fp16: bool
    cuda_version: str | None

    def best_precision(self) -> str:
        """Return the best supported precision string."""
        if self.supports_bf16:
            return "bf16"
        if self.supports_fp16:
            return "fp16"
        return "fp32"

    def to_dict(self) -> dict[str, Any]:
        """Serialize device info for logging/config."""
        return {
            "device": str(self.device),
            "device_name": self.device_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "cuda_version": self.cuda_version,
            "best_precision": self.best_precision(),
        }


def detect_device(prefer_cuda: bool = True) -> DeviceInfo:
    """Detect the best available compute device.

    Args:
        prefer_cuda: If True, use CUDA when available. Otherwise force CPU.

    Returns:
        DeviceInfo with details about the selected device.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = round(props.total_memory / (1024**3), 1)
        supports_bf16 = torch.cuda.is_bf16_supported()
        supports_fp16 = True  # All modern CUDA GPUs support fp16
        cuda_version = torch.version.cuda

        return DeviceInfo(
            device=device,
            device_name=props.name,
            gpu_memory_gb=gpu_memory_gb,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            cuda_version=cuda_version,
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        device_name="CPU",
        gpu_memory_gb=None,
        supports_bf16=hasattr(torch, "bfloat16"),
        supports_fp16=True,
        cuda_version=None,
    )


def get_precision_dtype(precision: str) -> torch.dtype:
    """Convert a precision string to a torch dtype.

    Args:
        precision: One of 'bf16', 'fp16', 'fp32'.

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If precision string is unrecognized.
    """
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if precision not in mapping:
        raise ValueError(f"Unknown precision '{precision}'. Expected one of: {list(mapping.keys())}")
    return mapping[precision]
