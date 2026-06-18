"""Tests for parameter counting utilities."""

import torch
import torch.nn as nn

from src.utils.params import count_parameters, count_parameters_by_module, format_param_table, ParamCount


class TestCountParameters:
    def test_simple_linear(self):
        model = nn.Linear(10, 5, bias=False)
        count = count_parameters(model)
        assert count.total == 50
        assert count.trainable == 50
        assert count.frozen == 0

    def test_with_bias(self):
        model = nn.Linear(10, 5, bias=True)
        count = count_parameters(model)
        assert count.total == 55  # 50 weight + 5 bias
        assert count.trainable == 55

    def test_frozen_parameters(self):
        model = nn.Linear(10, 5, bias=False)
        model.weight.requires_grad = False
        count = count_parameters(model)
        assert count.total == 50
        assert count.trainable == 0
        assert count.frozen == 50

    def test_nested_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20, bias=False),
            nn.Linear(20, 5, bias=False),
        )
        count = count_parameters(model)
        assert count.total == 10 * 20 + 20 * 5  # 300
        assert count.trainable == 300

    def test_millions_property(self):
        count = ParamCount(total=50_000_000, trainable=50_000_000, frozen=0)
        assert count.total_millions == 50.0
        assert count.trainable_millions == 50.0


class TestCountByModule:
    def test_breakdown(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 10, bias=False)

            def forward(self, x):
                return self.linear(self.embed(x))

        model = SimpleModel()
        counts = count_parameters_by_module(model)
        assert "embed" in counts
        assert "linear" in counts
        assert counts["embed"].total == 100 * 32
        assert counts["linear"].total == 32 * 10


class TestFormatParamTable:
    def test_produces_string(self):
        model = nn.Sequential(
            nn.Linear(10, 20, bias=False),
            nn.Linear(20, 5, bias=False),
        )
        table = format_param_table(model)
        assert isinstance(table, str)
        assert "TOTAL" in table
        assert "Trainable" in table


class TestParamCount:
    def test_to_dict(self):
        count = ParamCount(total=1_000_000, trainable=900_000, frozen=100_000)
        d = count.to_dict()
        assert d["total"] == 1_000_000
        assert d["total_millions"] == 1.0
        assert d["trainable_millions"] == 0.9

    def test_str(self):
        count = ParamCount(total=50_000_000, trainable=50_000_000, frozen=0)
        s = str(count)
        assert "50.00M" in s
