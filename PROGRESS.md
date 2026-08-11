# Progress

Driveway shot tracker build log. Milestones M0–M6 per the build plan.

## Status

| Milestone | State | Notes |
|---|---|---|
| M0 Repo + engine skeleton | **done** 2026-08-11 | acceptance test green; two adversarial review rounds applied |
| M1 Detector v0 integration | blocked on training + footage | detector v0 still training |
| M2 Shot logic | engine logic largely built in M0; acceptance pending owner's 50-shot clip | |
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
| 2026-08-11 ~14:25 | running |
| 2026-08-11 ~15:00 | running |
| 2026-08-11 ~15:40 | running |

## Waiting on owner (Aarnav)

- Driveway footage (3+ sessions, 60 fps landscape, rim in upper third) — needed before M1. **Critical path**: training will likely finish first.
- 50-shot ground-truth labels — needed for M2 acceptance.
- AA reference screenshots into `assets/reference/` — needed for M4.

## M0 summary

Engine skeleton complete: `ShotEngine` (frame-in/events-out; detector and pose
model are injected callables), `FrameState`/`ShotEvent` types matching the
plan's jsonl schema, single-ball tracker (motion gate, ≤4-frame gap
extrapolation, gated re-seeding), shot state machine (release streak, crossing
checks, rattle/occlusion policies, hard timeout), rim-width scale calibration
with drift recovery, metrics (entry angle, peak height, release velocity),
`VideoFileSource`, `CameraSource` interface stub, renderer stub, event log,
stats queries, upload pipeline wiring. 21 tests green, including the M0
acceptance test (scripted parabola → exactly one RESOLVED make).

Quality process: adversarial multi-agent review (35 agents) found 25 confirmed
defects in the first cut; a verification round on those fixes (3 agents +
attack simulations) found 11 more, including overcorrections. All fixed and
regression-tested; the verifiers' attack sims re-run clean.

Known limitations (documented decisions, revisit at M2/M6 with real footage):
- **2D rattle ambiguity**: a small pop-out that falls back in-span scores
  make/rattled; a pop above one rim width scores miss/rattled even if it drops
  back in. Mono 2D cannot separate these; every rattle is stamped
  `verdict_confidence=rattled` for the M2 audit, and M6 rim keypoints are the
  real fix.
- **One ball in frame** (plan v1 scope): a second resting ball near the hoop
  can capture the tracker after a track drop. ByteTrack swap noted in
  tracker docstring if real footage shows this.
- **Attempts that never reach the rim band** (apex below rim bottom) are
  discarded as noise per plan 6.2 — hopeless airballs are not logged.
- Duplicate/backwards timestamps can shift the detected release by a few
  frames (upload mode is immune: timestamps are synthesized frame_index/fps).
- Thresholds are in pixels (release speed, gates) and assume ~1080p-scale
  footage; M2 tuning on real clips should express them via calibration scale.

## Log

- **2026-08-11** — Repo initialized. `.gitignore` first commit (excludes `.claude/`, `AGENTS.md`, plan doc, `sessions/`, `.env`). Polled detector v0: running. Started M0.
- **2026-08-11** — M0 implemented; 9 tests green incl. acceptance test.
- **2026-08-11** — Adversarial review round 1 (35 agents): 25 confirmed findings — rattle double-counting, EMA-lag crossing gap, occlusion cap at tracker gap, gate/velocity spikes, extrapolated-sample pollution, test-quality gaps. All fixed; suite grew to 17 tests.
- **2026-08-11** — Verification round 2 (3 agents + attack sims): 11 findings incl. overcorrections from round 1 — ghost/gap-fabricated crossings, ungated re-seeding (phantom shots, wedged holds), rattle-out scored as make, release guard eating close-range shots, calibration lock-in on garbage, event-log truncation vs plan append. All fixed; suite at 21 tests; attack sims re-run clean. **M0 closed.**
- **2026-08-11** — Detector v0 still `running` at last poll (~15:40 UTC). Next: poll on next session/wake; M1 starts when training completes AND first footage arrives.
