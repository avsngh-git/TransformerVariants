# Skill Workflow Pattern — Code & Context Hygiene

## One-time setup

Run `/setup-matt-pocock-skills` first. It configures the issue tracker, triage labels, and doc layout that the other skills depend on.

## The core loop

**For every new idea:**

1. Start with `/grill-with-docs` — this is your quality gate. It sharpens the idea through a relentless interview and *persists what it learns* in `CONTEXT.md` and ADRs. This is where context hygiene begins: the domain model, decisions, and constraints get written down, not left floating in your head or in chat history.

2. Keep everything in one unbroken context window from grilling through to PRD/issues. This is the single most important context hygiene rule — the agent builds on its own reasoning without re-discovering it.

3. When you hit a question that conversation can't settle (state machines, tricky UI, business logic edge cases), `/handoff` → fresh session → `/prototype` → `/handoff` back. Prototypes are throwaway; the handoff document is the durable artifact.

4. For anything non-trivial, go `/to-prd` → `/to-issues`. Each issue becomes an independent unit of work with its own fresh session. This is where you *intentionally shed context* — each implementation session only carries the PRD + one issue, keeping it inside the smart zone.

## Context hygiene rules

- **Never compact mid-phase.** If you're mid-grill or mid-implementation, don't let the context window summarize away the details. Finish the phase first.
- **Use `/handoff` to fork, `/compact` to continue.** Handoff when you need a fresh session (prototype detours, switching between issues). Compact only at intentional phase boundaries within a single thread.
- **One issue per session for implementation.** This keeps each context window focused and well inside the ~120k token smart zone.
- **If a session is getting long and you haven't reached `/to-issues` yet**, don't push through degraded reasoning. `/handoff` and pick up in a new thread.

## Code quality layer

- Run `/improve-codebase-architecture` periodically between features. It surfaces "deepening opportunities" — places where the codebase can become more modular, testable, or AI-navigable. Each opportunity you pick becomes an idea that feeds back into the main flow via `/grill-with-docs`.
- Incoming bugs and requests go through `/triage` first, which produces agent-ready briefs. Don't skip this for external issues — the triage step ensures you understand the problem before you code the fix.

## The pattern

```
grill → decide → PRD → issues → [fresh session per issue] → implement
   ↑                                                              |
   └──── /improve-codebase-architecture (generates new ideas) ────┘
```

## Key insight

Context hygiene isn't about remembering everything — it's about knowing exactly when to forget. You preserve context within a phase (grilling, prototyping, implementing one issue) and deliberately shed it between phases using handoffs and fresh sessions. The durable artifacts (`CONTEXT.md`, ADRs, PRDs, issues) carry the *decisions*, so the next session doesn't need the *reasoning journey* that produced them.
