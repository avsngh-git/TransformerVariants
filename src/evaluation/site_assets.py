"""Portable model-internals assets for static sites.

The module deliberately emits data and images only.  It contains no HTML or
Jekyll assumptions, so a separate site repository can consume the bundle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.models.attention import CausalSelfAttention
from src.models.linear_attention import CausalLinearAttention
from src.models.modern_attention import ModernAttention
from src.models.rope import apply_rope


@dataclass(frozen=True)
class SiteAssetResult:
    """Paths produced by :func:`export_site_assets`."""

    output_dir: Path
    manifest_path: Path
    model_internals_path: Path
    attention_patterns_path: Path | None
    plot_paths: tuple[Path, ...]


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats and tensors into strict JSON values."""
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 7)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    safe_payload = _json_safe(payload)
    if compact:
        serialized = json.dumps(safe_payload, separators=(",", ":"), allow_nan=False)
    else:
        serialized = json.dumps(safe_payload, indent=2, allow_nan=False)
    path.write_text(serialized + "\n", encoding="utf-8")


def _asset_slug(value: Any) -> str:
    """Return a traversal-safe, collision-stable identifier for generated files."""
    text = str(value)
    if not text or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in text
    ):
        raise ValueError(f"Variant {text!r} is not a safe asset identifier")
    return text


def _attention_pattern_entropy(capture: dict[str, Any]) -> dict[str, Any]:
    """Summarize one captured context's entropy by layer and head."""
    per_head: list[list[float]] = []
    per_layer: list[float] = []
    for layer in capture.get("layers", []):
        head_values = []
        for head in layer.get("heads", []):
            weights = torch.tensor(head["weights"], dtype=torch.float64)
            terms = torch.where(weights > 0, weights * weights.log(), torch.zeros_like(weights))
            head_values.append(float(-terms.sum(dim=-1).mean()))
        per_head.append(head_values)
        per_layer.append(sum(head_values) / len(head_values) if head_values else 0.0)
    return {
        "source": "reconstructed_pre_dropout_softmax",
        "context_length": len(capture.get("tokens", [])),
        "per_layer": per_layer,
        "per_head": per_head,
    }


def _load_model_internals(report_dir: Path) -> dict[str, Any]:
    metrics_path = report_dir / "raw" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    aggregated = metrics.get("probes", {}).get("aggregated")
    if not isinstance(aggregated, dict):
        raise ValueError(f"Missing probes.aggregated object in {metrics_path}")

    variants: dict[str, Any] = {}
    for name, probe in sorted(aggregated.items()):
        if not isinstance(probe, dict):
            continue
        variants[name] = {
            "n_seeds": probe.get("n"),
            "stable_rank": probe.get("stable_rank"),
            "cka": probe.get("cka"),
            "attention_entropy": probe.get("attention_entropy"),
        }
    return {
        "schema_version": 1,
        "source": str(metrics_path),
        "variants": variants,
    }


def _plot_model_internals(internals: dict[str, Any], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific dependency error
        raise RuntimeError("PNG export requires the project's 'viz' dependencies") from exc

    variants = internals["variants"]
    paths: list[Path] = []
    plot_specs = [
        ("stable_rank", "per_layer", "Stable rank by layer", "Stable rank"),
        ("cka", "adjacent_curve", "Adjacent-layer CKA", "Linear CKA"),
        ("attention_entropy", "per_layer", "Attention entropy by layer", "Entropy (nats)"),
        ("attention_pattern_entropy", "per_layer", "Captured attention entropy", "Entropy (nats)"),
    ]
    for metric, field, title, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plotted = False
        for name, payload in variants.items():
            metric_payload = payload.get(metric)
            values = metric_payload.get(field) if isinstance(metric_payload, dict) else None
            if not values:
                continue
            x = list(range(len(values)))
            ax.plot(x, values, marker="o", markersize=3, linewidth=1.5, label=name)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel="Layer" if metric != "cka" else "Layer pair", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        path = output_dir / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _attention_weights(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reconstruct the exact pre-dropout softmax probabilities for one layer."""
    batch, length, _ = x.shape
    if isinstance(module, CausalSelfAttention):
        q, k, _ = module.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch, length, module.n_head, module.d_head).transpose(1, 2)
        k = k.view(batch, length, module.n_head, module.d_head).transpose(1, 2)
    elif isinstance(module, ModernAttention):
        q, k, _ = module.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch, length, module.n_head, module.d_head).transpose(1, 2)
        k = k.view(batch, length, module.n_head, module.d_head).transpose(1, 2)
        q = apply_rope(q, module.rope_cos[:length], module.rope_sin[:length])
        k = apply_rope(k, module.rope_cos[:length], module.rope_sin[:length])
    elif hasattr(module, "_project_qkv") and hasattr(module, "_apply_position"):
        q, k, _ = module._project_qkv(x, batch, length)
        q, k = module._apply_position(q, k, offset=0)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if k.size(1) != q.size(1):
            if q.size(1) % k.size(1) != 0:
                raise ValueError("Query heads must be divisible by key/value heads")
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    else:
        raise TypeError(f"Unsupported attention module: {type(module).__name__}")

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (module.d_head**-0.5)
    positions = torch.arange(length, device=x.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)

    slopes = getattr(module, "alibi_slopes", None)
    if slopes is not None:
        distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
        scores = scores - slopes.float().view(1, -1, 1, 1) * distance.view(1, 1, length, length)

    window_size = getattr(module, "_window_size", None)
    if window_size is None:
        window_size = getattr(getattr(module, "config", None), "window_size", None)
    if window_size is not None:
        causal = causal & ((positions.unsqueeze(1) - positions.unsqueeze(0)) <= window_size)

    scores = scores.masked_fill(~causal.view(1, 1, length, length), float("-inf"))
    return F.softmax(scores, dim=-1)


def capture_attention_patterns(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    token_labels: Sequence[str] | None = None,
    variant: str | None = None,
    layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Capture all heads for selected layers without changing model forwards.

    Flash/SDPA implementations intentionally avoid materializing an O(T²)
    matrix.  This function reconstructs their mathematically equivalent
    pre-dropout softmax probabilities from the layer input and projections.
    Causal linear attention is reported as unsupported because its recurrent
    feature-map computation has no conventional pairwise softmax matrix.
    """
    if token_ids.ndim != 2 or token_ids.size(0) != 1:
        raise ValueError("token_ids must have shape (1, sequence_length)")
    if token_ids.size(1) < 1:
        raise ValueError("attention capture requires at least one token")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("model must expose a blocks sequence")
    if any(isinstance(block.attn, CausalLinearAttention) for block in blocks):
        return {
            "variant": variant or getattr(getattr(model, "config", None), "variant", "unknown"),
            "status": "unsupported",
            "reason": (
                "Causal linear attention uses recurrent feature-map statistics and does not "
                "define a conventional pairwise softmax attention matrix."
            ),
        }

    selected = list(range(len(blocks))) if layers is None else list(layers)
    if not selected or any(index < 0 or index >= len(blocks) for index in selected):
        raise ValueError("layers must contain valid model layer indices")

    captured_inputs: dict[int, torch.Tensor] = {}
    handles = []
    for index in selected:

        def save_input(_module, args, *, layer_index=index):
            captured_inputs[layer_index] = args[0].detach()

        handles.append(blocks[index].attn.register_forward_pre_hook(save_input))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(token_ids)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    labels = (
        list(token_labels) if token_labels is not None else [str(x) for x in token_ids[0].tolist()]
    )
    if len(labels) != token_ids.size(1):
        raise ValueError("token_labels length must match sequence length")

    layer_payloads = []
    with torch.no_grad():
        for index in selected:
            weights = _attention_weights(blocks[index].attn, captured_inputs[index])[0].cpu()
            layer_payloads.append(
                {
                    "layer": index,
                    "mean_weights": _json_safe(weights.mean(dim=0)),
                    "heads": [
                        {"head": head, "weights": _json_safe(weights[head])}
                        for head in range(weights.size(0))
                    ],
                }
            )

    return {
        "variant": variant or getattr(getattr(model, "config", None), "variant", "unknown"),
        "status": "supported",
        "method": "reconstructed_pre_dropout_softmax",
        "token_ids": token_ids[0].detach().cpu().tolist(),
        "tokens": labels,
        "layers": layer_payloads,
    }


def _plot_attention_patterns(patterns: dict[str, Any], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PNG export requires the project's 'viz' dependencies") from exc

    paths: list[Path] = []
    for capture in patterns.get("variants", []):
        if capture.get("status") != "supported":
            continue
        labels = capture["tokens"]
        for layer in capture["layers"]:
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            image = ax.imshow(layer["mean_weights"], cmap="magma", vmin=0.0, aspect="auto")
            ax.set(
                title=f"{capture['variant']} — layer {layer['layer']} (mean across heads)",
                xlabel="Key token",
                ylabel="Query token",
            )
            if len(labels) <= 32:
                ticks = list(range(len(labels)))
                ax.set_xticks(ticks, labels, rotation=90, fontsize=6)
                ax.set_yticks(ticks, labels, fontsize=6)
            fig.colorbar(image, ax=ax, label="Attention probability")
            path = (
                output_dir
                / f"attention_{_asset_slug(capture['variant'])}_layer_{layer['layer']:02d}.png"
            )
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def export_site_assets(
    report_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    attention_patterns: Sequence[dict[str, Any]] | None = None,
) -> SiteAssetResult:
    """Export strict JSON and PNG assets suitable for a separate static site."""
    report_path = Path(report_dir)
    destination = Path(output_dir) if output_dir is not None else report_path / "site_assets"

    captures = list(attention_patterns) if attention_patterns is not None else None
    variant_slugs: dict[str, str] = {}
    if captures is not None:
        used_slugs: set[str] = set()
        for capture in captures:
            if capture.get("status") != "supported":
                continue
            variant = str(capture.get("variant", "unknown"))
            slug = _asset_slug(variant)
            if slug in used_slugs:
                raise ValueError(f"Duplicate attention asset identifier: {slug}")
            used_slugs.add(slug)
            variant_slugs[variant] = slug

    destination.mkdir(parents=True, exist_ok=True)

    internals = _load_model_internals(report_path)
    if captures is not None:
        for capture in captures:
            if capture.get("status") != "supported":
                continue
            variant = str(capture.get("variant", "unknown"))
            payload = internals["variants"].setdefault(
                variant,
                {
                    "n_seeds": None,
                    "stable_rank": None,
                    "cka": None,
                    "attention_entropy": None,
                },
            )
            payload["attention_pattern_entropy"] = _attention_pattern_entropy(capture)
    internals_path = destination / "model_internals.json"
    _write_json(internals_path, internals)
    plot_paths = _plot_model_internals(internals, destination)

    attention_path: Path | None = None
    attention_variant_paths: list[Path] = []
    if captures is not None:
        index_entries = []
        for capture in captures:
            entry = {
                "variant": capture.get("variant", "unknown"),
                "status": capture.get("status", "error"),
            }
            if "checkpoint_dir" in capture:
                entry["checkpoint_dir"] = capture["checkpoint_dir"]
            if "context" in capture:
                entry["context"] = capture["context"]
            if capture.get("status") == "supported":
                slug = variant_slugs[str(entry["variant"])]
                variant_path = destination / f"attention_patterns_{slug}.json"
                _write_json(variant_path, capture, compact=True)
                attention_variant_paths.append(variant_path)
                entry.update(
                    {
                        "asset": variant_path.name,
                        "tokens": capture.get("tokens", []),
                        "token_ids": capture.get("token_ids", []),
                        "layers": [layer.get("layer") for layer in capture.get("layers", [])],
                        "head_count": len(capture.get("layers", [{}])[0].get("heads", []))
                        if capture.get("layers")
                        else 0,
                    }
                )
            else:
                entry["reason"] = capture.get("reason", "capture unavailable")
            index_entries.append(entry)
        attention_payload = {"schema_version": 1, "variants": index_entries}
        attention_path = destination / "attention_patterns.json"
        _write_json(attention_path, attention_payload)
        plot_paths.extend(
            _plot_attention_patterns({"schema_version": 1, "variants": captures}, destination)
        )

    assets: dict[str, Any] = {
        "model_internals": internals_path.name,
        "plots": [path.name for path in plot_paths],
    }
    if attention_path is not None:
        assets["attention_patterns"] = attention_path.name
        assets["attention_variants"] = [path.name for path in attention_variant_paths]
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frontend": "agnostic",
        "source_report": str(report_path),
        "assets": assets,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return SiteAssetResult(
        output_dir=destination,
        manifest_path=manifest_path,
        model_internals_path=internals_path,
        attention_patterns_path=attention_path,
        plot_paths=tuple(plot_paths),
    )
