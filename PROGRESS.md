# Progress

Driveway shot tracker build log. Milestones M0–M6 per the build plan.

## Status

| Milestone | State | Notes |
|---|---|---|
| M0 Repo + engine skeleton | in progress | started 2026-08-11 |
| M1 Detector v0 integration | blocked on training + footage | detector v0 still training |
| M2 Shot logic | not started | needs owner's 50-shot ground-truth clip |
| M3 Pose | not started | |
| M4 Renderer | not started | needs AA reference screenshots in assets/reference/ |
| M5 Modes | not started | |
| M6 Precision + polish | not started | |

## Detector v0 (bootstrap, training on Roboflow — poll, never retrain)

Model ID: `aarnavs-space/basketball-shooting-robot-kbsro-1-rfdetr-small-t1`
Project/version: `aarnavs-space/basketball-shooting-robot-kbsro` v1, training ID `4fa4d9f6b5a838b80cf2`.

| Polled (UTC) | Status |
|---|---|
| 2026-08-11 ~14:00 | running (launched 2026-08-11 13:34) |

## Waiting on owner (Aarnav)

- Driveway footage (3+ sessions, 60 fps landscape, rim in upper third) — needed before M1.
- 50-shot ground-truth labels — needed for M2 acceptance.
- AA reference screenshots into `assets/reference/` — needed for M4.

## Log

- **2026-08-11** — Repo initialized. `.gitignore` first commit (excludes `.claude/`, `AGENTS.md`, plan doc, `sessions/`, `.env`). Polled detector v0: running. Starting M0: engine skeleton, config system, sources, renderer stub, synthetic-parabola test.
