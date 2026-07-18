# ADR 0005: Health Monitor as Injected Trainer Dependency

## Status

Accepted (2026-07-04)

## Context

Phase 11 adds a training health monitor that detects anomalies (NaN, gradient
spikes, loss spikes) and triggers recovery actions (skip step, rollback to
checkpoint). The monitor needs to intervene in the training loop — it doesn't
just observe, it changes control flow.

Two integration patterns were considered:

1. **Direct injection** — `HealthMonitor` is an optional constructor argument to
   `Trainer`. The training loop calls `monitor.check(step, loss, grad_norm)` and
   acts on the returned `Action` enum.

2. **Callback/hook system** — `Trainer` emits events (step_completed, etc.) and
   the monitor subscribes. More decoupled, requires building a callback
   infrastructure that doesn't exist today.

## Decision

We chose direct injection (Option 1).

## Consequences

**Benefits:**
- Follows the existing pattern in `Trainer` (logger, dataloader are already injected)
- The monitor can *intervene* (SKIP_STEP, ROLLBACK), not just observe — this maps
  naturally to a synchronous call that returns an action
- No callback infrastructure to build and maintain
- Easy to test: mock the monitor, verify Trainer respects actions
- `monitor=None` is the default — zero behavioral change for existing code
- Fault-tolerant training seeds a verified step-zero checkpoint, builds a ten-sample
  finite baseline before z-score decisions, and retries rather than advances a
  restored step. NaN/Inf detection remains immediate.

**Tradeoffs:**
- If future hooks are needed (e.g., logging callbacks, profiling), they'd be
  separate arguments rather than a unified event system
- Tighter coupling between Trainer and monitor interface

**Mitigations:**
- The `HealthMonitor` interface is minimal (one method: `check() → Action`)
- If a callback system is ever needed, the monitor can become one of multiple
  listeners — the refactor is straightforward
- For a single-GPU training project, the callback system would be over-engineering
