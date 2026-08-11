# Progress

Driveway shot tracker build log. Milestones M0–M6 per the build plan.

## Status

| Milestone | State | Notes |
|---|---|---|
| M0 Repo + engine skeleton | **done** 2026-08-11 | acceptance test green; two adversarial review rounds applied |
| M1 Detector v0 integration | **done** 2026-08-11 | ball-in-flight 98.1% (target 90) PASS; rim IoU gap logged as M6 fine-tune input |
| M2 Shot logic | **23/23 on owner-labeled session** (pending adversarial verification round) | owner substituted 23 labeled clips for the 50-shot clip; eval set = tuning set, see caveat |
| M3 Pose | not started | needed to fix release-timing accuracy (wrist rule) |
| M4 Renderer | not started | reference layout documented from owner's screenshots |
| M5 Modes | not started | |
| M6 Precision + polish | not started | fine-tune inputs logged below |

## Detector v0 (bootstrap, training on Roboflow — poll, never retrain)

Model ID: `aarnavs-space/basketball-shooting-robot-kbsro-1-rfdetr-small-t1`
Project/version: `aarnavs-space/basketball-shooting-robot-kbsro` v1, training ID `4fa4d9f6b5a838b80cf2`.

| Polled (UTC) | Status |
|---|---|
| 2026-08-11 ~14:00 | running (launched 2026-08-11 13:34) |
| 2026-08-11 ~14:25 | running |
| 2026-08-11 ~15:00 | running |
| 2026-08-11 ~15:40 | running |
| 2026-08-11 ~16:25 | **finished** (ended 16:21 UTC) — mAP50 88.84, precision 75.7, recall 84.9 |

## Waiting on owner (Aarnav)

- **50-shot ground-truth labels** — the M2 acceptance gate. First session (17
  clips) is in; a labeled ~50-shot session (60 fps please) is the next input.
- ~~Confirm 7103 / 7097~~ — answered 2026-08-11: 7103 is a real (missed)
  attempt the engine failed to segment; 7097 is a genuine rattle-in make
  (engine verdict was correct).
- Future sessions: shoot at **60 fps** (current clips are 30 fps; crossing
  frames are sparse and soft launches blur below the release threshold).
- Homography clicks + look sign-off at M4.

## M1 summary (2026-08-11)

Detector v0 loaded by model ID via pinned `inference==1.3.10`, running
locally (ONNX). Full pipeline (VideoFileSource → ShotEngine → EventLog) ran
on all 17 clips of session-2026-08-11 (~62s total, 30 fps).

- **Ball detected in flight: 98.1% (759/774 frames) — PASSES the ≥90% bar.**
- **Rim: detected on ~100% of frames, but box unstable**: only 44.1% of
  detections reach IoU ≥0.8 vs the session median (median IoU 0.776 vs 0.8
  target). Cause measured on footage: box height flaps with net/backboard
  inclusion; width is stable within a few px. Per plan, this gap is an input
  to the **M6 fine-tune** (oversample rim+net frames, inconsistent rim box
  annotations in the source dataset), not a reason to retrain the bootstrap.
  Engine compensates: shot geometry uses the median box; calibration drift
  detection judges center+width only.
- Verdicts (owner ground truth, confirmed 2026-08-11): **16/16 segmented
  attempts scored correctly** — 7101 make/clean ✓, 7097 make/rattled ✓ (a
  real rattle-in: "bounced on the rim a couple times and then made it" — the
  rattle policy called it right on real footage), 14 misses ✓. The one miss
  of the milestone is segmentation, not verdicts: 7103 (below).
- **7103 = confirmed missed attempt** (owner: real attempt, no make). Root
  cause from per-frame data: soft short-range launch measured at only
  2.2–2.7 m/s smoothed (30 fps blur + detection gap at launch diluting the
  span-averaged velocity); the 3-frame streak died 23 px/s short of the
  2.5 m/s threshold. Windups measure 2.0–3.6 m/s, so velocity alone cannot
  separate soft launches from windups at 30 fps. Fix path: M3's
  wrist-separation release rule (lower the velocity floor once ball-in-hand
  frames are excluded by pose), plus 60 fps footage. Launch-window frames
  with the ball in hands also go on the M6 auto-label list (blur/occlusion
  hard case).
- Engine changes driven by real footage: release threshold expressed in m/s
  via calibrated scale (2.5 m/s; the ball-raise into the pocket was arming
  releases at 200 px/s), calibration drift check tolerant of height flap, and
  `ShotEngine.finalize()` — clip-per-shot footage cuts at the outcome, so
  end-of-stream now resolves open shots (occluded-confidence) instead of
  dropping them.
- Release velocities after fixes: 3.6–6.6 m/s (plausible); release *timing*
  is still velocity-only and will tighten at M3 with the wrist-separation
  rule.

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

## M2 summary (2026-08-11)

Owner declined recording a fresh 50-shot session and substituted full labels
for session-2026-08-11 (23 clips, one attempt each; makes: 7090, 7097, 7101,
7109 — recorded in `scripts/groundtruth/session-2026-08-11.json`).

**Scorecard: 23/23 — zero segmentation errors, zero verdict errors**
(`scripts/m2_eval.py`; proportional bar was ≥21/23). The two labels that
initially failed drove real fixes:

- 7090/7109 (makes scored miss): the rim-top crossing hid inside a 1–2 frame
  detection gap — net occlusion at the crossing, plan 6.3's exact case. Fix:
  crossings may bridge a ≤4-frame gap between *observed* endpoints that hug
  the rim (last seen ≤2 rim-heights above the top, reappearing ≤1 rim-height
  below the bottom, interpolated x-cross strictly in-span), flagged
  `occluded`. Behind-the-rim airball attacks from the M0 review still resolve
  correctly (re-verified).
- 7103 (soft launch unsegmented): release floor lowered to 2.0 m/s. Windups
  reaching 2.0–3.6 m/s may arm early — this skews release *timing/metrics*
  on some shots, never verdicts, until M3's wrist-separation rule.

Caveats, recorded honestly: the 23 clips served as both tuning set and eval
set (the plan's fresh-clip design avoids this); the session is 30 fps where
the plan asked for 60; and n=23 < 50. M6's re-eval (≥48/50 equivalent) should
use fresh footage. Adversarial verification round on the M1/M2-era engine
changes launched before closing the milestone.

## Log

- **2026-08-11** — Repo initialized. `.gitignore` first commit (excludes `.claude/`, `AGENTS.md`, plan doc, `sessions/`, `.env`). Polled detector v0: running. Started M0.
- **2026-08-11** — M0 implemented; 9 tests green incl. acceptance test.
- **2026-08-11** — Adversarial review round 1 (35 agents): 25 confirmed findings — rattle double-counting, EMA-lag crossing gap, occlusion cap at tracker gap, gate/velocity spikes, extrapolated-sample pollution, test-quality gaps. All fixed; suite grew to 17 tests.
- **2026-08-11** — Verification round 2 (3 agents + attack sims): 11 findings incl. overcorrections from round 1 — ghost/gap-fabricated crossings, ungated re-seeding (phantom shots, wedged holds), rattle-out scored as make, release guard eating close-range shots, calibration lock-in on garbage, event-log truncation vs plan append. All fixed; suite at 21 tests; attack sims re-run clean. **M0 closed.**
- **2026-08-11** — Detector v0 still `running` at last poll (~15:40 UTC). Next: poll on next session/wake; M1 starts when training completes AND first footage arrives.
- **2026-08-11 (pm)** — Training finished (mAP50 88.84). Owner delivered 17 clips (30 fps) into `footage/session-2026-08-11/` and the AA reference screenshots (transcribed to `assets/reference/aa-reference-layout.md`). Wired RFDETRDetector through pinned inference runtime; ran all clips; fixed calibration drift thrash, early-release arming, and added end-of-stream finalize. **M1 closed**: ball 98.1% PASS; rim box stability shortfall documented as M6 fine-tune input. Owner question noted: Roboflow Workflows deliberately not used — plan architecture injects the detector as a local callable into our stateful engine ("free and local" product angle; Workflows wrap only the detection block and add a serving hop).
