"""Tests for device detection utilities."""

import torch
import pytest

from src.utils.device import detect_device, get_precision_dtype, DeviceInfo


class TestDetectDevice:
    def test_returns_device_info(self):
        info = detect_device()
        assert isinstance(info, DeviceInfo)
        assert info.device in (torch.device("cuda"), torch.device("cpu"))

    def test_force_cpu(self):
        info = detect_device(prefer_cuda=False)
        assert info.device == torch.device("cpu")
        assert info.device_name == "CPU"
        assert info.gpu_memory_gb is None

    def test_best_precision_returns_valid(self):
        info = detect_device()
        assert info.best_precision() in ("bf16", "fp16", "fp32")

    def test_to_dict_serializable(self):
        info = detect_device()
        d = info.to_dict()
        assert isinstance(d, dict)
        assert "device" in d
        assert "best_precision" in d


class TestGetPrecisionDtype:
    def test_bf16(self):
        assert get_precision_dtype("bf16") == torch.bfloat16

    def test_fp16(self):
        assert get_precision_dtype("fp16") == torch.float16

    def test_fp32(self):
        assert get_precision_dtype("fp32") == torch.float32

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            get_precision_dtype("fp64")
