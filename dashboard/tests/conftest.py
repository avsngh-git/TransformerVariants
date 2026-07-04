"""Shared pytest fixtures and Hypothesis strategies for dashboard tests."""

import json
import pathlib

import pytest
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_POOL = [
    "vanilla",
    "modern",
    "alibi",
    "gqa",
    "swa",
    "swa_interleaved",
    "linear",
]

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def variant_names_strategy(draw, min_size=1, max_size=7):
    """Generate a random list of 1–7 unique variant names from the realistic pool."""
    names = draw(
        st.lists(
            st.sampled_from(VARIANT_POOL),
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    return sorted(names)


@st.composite
def metrics_dict_strategy(draw):
    """Generate a random metrics dict for a single seed entry."""
    val_loss = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    perplexity = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    icl_exponent = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    r_squared = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    A = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    alpha = icl_exponent
    C = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))

    return {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "icl_exponent": icl_exponent,
        "icl_fit_params": {
            "A": A,
            "alpha": alpha,
            "C": C,
            "r_squared": r_squared,
        },
    }


@st.composite
def aggregated_data_strategy(draw, variant_names=None):
    """Generate random aggregated data for a set of variants.

    Returns a dict mapping variant_name → {metric_name: {mean, std}}.
    """
    if variant_names is None:
        variant_names = draw(variant_names_strategy())

    aggregated = {}
    for name in variant_names:
        val_loss_mean = draw(st.floats(min_value=1.0, max_value=8.0, allow_nan=False, allow_infinity=False))
        val_loss_std = draw(st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False))
        perplexity_mean = draw(st.floats(min_value=2.0, max_value=500.0, allow_nan=False, allow_infinity=False))
        perplexity_std = draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
        step_flops_mean = draw(st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False))
        icl_alpha_mean = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
        icl_alpha_std = draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False))

        aggregated[name] = {
            "val_loss": {"mean": val_loss_mean, "std": val_loss_std},
            "perplexity": {"mean": perplexity_mean, "std": perplexity_std},
            "step_flops": {"mean": step_flops_mean, "std": None},
            "icl_alpha": {"mean": icl_alpha_mean, "std": icl_alpha_std},
        }

    return aggregated


@st.composite
def learning_curve_data_strategy(draw, num_steps=None):
    """Generate random learning curve data (list of log entries) for one seed."""
    if num_steps is None:
        num_steps = draw(st.integers(min_value=5, max_value=50))

    entries = []
    for i in range(num_steps):
        step = (i + 1) * 100
        tokens_seen = step * 512
        wallclock = draw(st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False))
        cumulative_flops = draw(st.floats(min_value=1e6, max_value=1e15, allow_nan=False, allow_infinity=False))
        val_loss = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))

        entries.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "wallclock": wallclock,
            "cumulative_flops": cumulative_flops,
            "val_loss": val_loss,
        })

    return entries


@st.composite
def probe_data_strategy(draw):
    """Generate random probe data for a single variant seed."""
    num_layers = draw(st.integers(min_value=2, max_value=12))
    num_distances = draw(st.integers(min_value=2, max_value=10))

    mqar_accuracies = [
        draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for _ in range(num_distances)
    ]
    overall_mqar = sum(mqar_accuracies) / len(mqar_accuracies) if mqar_accuracies else 0.0

    stable_rank_per_layer = [
        draw(st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
        for _ in range(num_layers)
    ]
    stable_rank_mean = sum(stable_rank_per_layer) / len(stable_rank_per_layer)
    stable_rank_std = draw(st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False))

    cka_adjacent = [
        draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for _ in range(num_layers - 1)
    ]
    # Full CKA matrix (symmetric, diagonal = 1.0)
    cka_matrix = [[0.0] * num_layers for _ in range(num_layers)]
    for i in range(num_layers):
        cka_matrix[i][i] = 1.0
        for j in range(i + 1, num_layers):
            val = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
            cka_matrix[i][j] = val
            cka_matrix[j][i] = val

    attention_entropy_per_layer = [
        draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        for _ in range(num_layers)
    ]
    attention_entropy_mean = sum(attention_entropy_per_layer) / len(attention_entropy_per_layer)

    return {
        "mqar": {
            "distances": list(range(1, num_distances + 1)),
            "accuracies": mqar_accuracies,
            "overall_accuracy": overall_mqar,
        },
        "stable_rank": {
            "per_layer": stable_rank_per_layer,
            "mean": stable_rank_mean,
            "std": stable_rank_std,
        },
        "cka": {
            "adjacent_similarities": cka_adjacent,
            "full_matrix": cka_matrix,
        },
        "attention_entropy": {
            "per_layer": attention_entropy_per_layer,
            "mean": attention_entropy_mean,
        },
    }


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_dir():
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def valid_metrics_path():
    """Return the path to the valid metrics fixture file."""
    return FIXTURES_DIR / "valid_metrics.json"


@pytest.fixture
def malformed_metrics_path():
    """Return the path to the malformed JSON fixture file."""
    return FIXTURES_DIR / "malformed.json"


@pytest.fixture
def partial_variants_path():
    """Return the path to the partial variants fixture file."""
    return FIXTURES_DIR / "partial_variants.json"


@pytest.fixture
def valid_metrics_data(valid_metrics_path):
    """Load and return the valid metrics fixture as a dict."""
    with open(valid_metrics_path) as f:
        return json.load(f)


@pytest.fixture
def partial_variants_data(partial_variants_path):
    """Load and return the partial variants fixture as a dict."""
    with open(partial_variants_path) as f:
        return json.load(f)
