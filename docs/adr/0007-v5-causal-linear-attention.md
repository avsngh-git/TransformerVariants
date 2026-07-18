# ADR 0007: V5 Uses Causal Feature-Map Linear Attention

## Status

Accepted

## Context

The main-scale V5 evaluation reported validation loss `0.0079`, while the
other decoder variants were near `3.3–3.9`. A suffix-perturbation test showed
that changing future tokens changed earlier V5 outputs.

The Linformer implementation projected all keys and values across the full
sequence before computing per-query attention. Every projected slot therefore
contained future-token information. Because projected slots do not retain a
token-order identity, applying a triangular mask after projection cannot make
this construction causal.

This violates the decoder-only next-token-prediction contract. All existing
Linformer V5 training and evaluation artifacts are non-comparable.

## Decision

V5 is **causal linear attention**, following the feature-map formulation from
Katharopoulos et al., [Transformers are RNNs](https://arxiv.org/abs/2006.16236):

- `phi(x) = ELU(x) + 1` produces positive query and key features.
- Each position reads only prefix key/value statistics through cumulative
  state, so position `i` depends only on positions `j <= i`.
- RoPE is applied to queries and keys before the feature map for positional
  parity with the modern baseline.
- Training evaluates the exact recurrence in chunks. A triangular local
  matrix handles the current chunk and accumulated state handles earlier
  chunks; this is algebraically equivalent to the token-wise recurrence.

The implementation remains under the registry key `linear` and variant ID V5,
but it must not be described as Linformer.

## Consequences

- V5 attention complexity is `O(T * d_head^2)`, linear in sequence length.
- Attention entropy is unavailable because no softmax probability matrix is
  materialized.
- Recurrent generation state is exposed as fixed-size numerator/denominator prefix statistics
  through the project's shared KV-cache interface.
- The three V5 main-scale seeds must be retrained, followed by a full evaluation
  rerun. Old V5 checkpoints and their rows/plots must not be reused.
- A shared future-token leakage suite now covers every unique attention path.
- ADRs 0002 and 0003 are superseded by this decision.
