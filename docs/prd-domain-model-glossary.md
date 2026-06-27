# PRD: Domain Model Glossary and Documentation Restructuring

**Label:** `ready-for-agent`

---

## Problem Statement

The project's documentation conflated two responsibilities in a single file (`docs/CONTEXT.md`): domain vocabulary (what terms mean) and operational state (what's built, how to run things). This made it unclear what the canonical definition of core terms like "variant," "sub-variant," "scale," and "run" actually were. Agents and humans reading the project had no authoritative glossary to reference, leading to inconsistent terminology across code, docs, and discussions.

## Solution

Split the documentation into two purpose-specific files with clear, non-overlapping responsibilities:

- **`CONTEXT.md`** (project root) — A pure domain glossary defining the ubiquitous language. Contains canonical definitions of: Model, Variant, Sub-variant, Architectural Component, Compute Optimization, Scale, Run, Experiment, Shard, Residual Stream, Teacher Forcing, Token Budget, Next-Token Prediction, Mixed Precision, Gradient Accumulation. Also captures invariants and open questions.

- **`docs/STATUS.md`** — Operational project state. What phases are complete, training runs, datasets, config files, how to run tests, file tree. Updated after each phase.

## User Stories

1. As a developer returning to the project after time away, I want a single authoritative glossary of domain terms, so that I know exactly what "variant" vs "sub-variant" vs "scale" means without ambiguity.
2. As an AI agent about to modify the codebase, I want to read one file for domain vocabulary and a different file for operational state, so that I don't confuse "what terms mean" with "what's been built."
3. As a developer naming a new module, I want to check CONTEXT.md for the canonical term, so that my naming is consistent with the rest of the project.
4. As a developer onboarding to the project, I want STATUS.md to tell me what's implemented and how to run it, so that I can get productive quickly.
5. As a developer adding a new variant, I want the glossary to clearly define what makes something a variant vs a sub-variant, so that I classify my work correctly.
6. As a developer designing the evaluation framework, I want the glossary to distinguish "run" from "experiment," so that I model the data correctly.
7. As a developer working on the data pipeline, I want a precise definition of "shard," so that I don't confuse it with epochs or document boundaries.
8. As a developer reviewing training configs, I want "token budget" and "gradient accumulation" defined precisely, so that I configure experiments correctly.
9. As a future collaborator reading the project, I want the glossary to note open questions (like the evaluation axis), so that I know what's unresolved.
10. As a developer writing documentation, I want to reference CONTEXT.md to ensure I use terms consistently, so that the project's language stays coherent.
11. As a developer updating project state after finishing a phase, I want a dedicated STATUS.md to update, so that I don't accidentally alter domain definitions.
12. As an AI agent running the domain-modeling skill, I want CONTEXT.md at the project root in the expected format, so that the skill can read and update it correctly.

## Implementation Decisions

- **Two-file split**: CONTEXT.md is the glossary (root level, rarely changes), STATUS.md is operational state (docs/ directory, changes every phase). No overlap in content.
- **CONTEXT.md format**: Grouped into "Core Concepts," "Training & Data Concepts," "Invariants," and "Open Questions." Each term has a one-paragraph canonical definition.
- **Variant taxonomy**: A variant is a coherent design philosophy (published recipe) that may include architectural components, new mechanisms, compute optimizations, or any combination. A sub-variant is a single component swap within a variant's recipe.
- **Scale semantics**: debug is never compared formally. main and stretch are benchmark scales. Dimensions are fixed per scale; parameter count is whatever falls out (±5% tolerance across variants).
- **agents/domain.md updated**: Reflects that both CONTEXT.md and STATUS.md now exist.
- **Old docs/CONTEXT.md deleted**: Content fully migrated to the two new files.

## Testing Decisions

- **Seam**: The CONTEXT.md glossary is the single source of domain truth. The "test" is consistency — terms used in code, docs, and discussion must align with CONTEXT.md definitions.
- **No code-level tests**: This is a documentation restructuring. Validation is human/agent review — does STATUS.md accurately reflect what exists in src/, tests/, configs/?
- **Prior art**: The project already validates agent behavior against docs/agents/domain.md — this extends that pattern.

## Out of Scope

- Evaluation framework design (explicitly deferred until all variants are implemented)
- ADR creation (no architectural decisions were made that meet the three-criteria bar: hard to reverse, surprising, real trade-off)
- Changes to any source code, configs, or test files
- Updating the README phase table (still shows phases as "Planned" — that's a separate maintenance task)

## Further Notes

- The open question about evaluation axis (fixed compute vs fixed data vs Pareto) is recorded in CONTEXT.md and should be resolved during Phase 8+ when the evaluation framework is designed.
- CONTEXT.md should be updated inline during future grilling sessions whenever a term is resolved — don't batch updates.
