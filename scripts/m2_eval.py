"""M2 scorecard: engine verdicts vs owner ground truth.

Each clip holds exactly one attempt. A clip scores correct when the engine
emits exactly one event whose verdict matches the label; zero events
(unsegmented) or extra events (split attempt) score wrong. M2 acceptance is
>= 90% correct with every attempt segmented (plan: 45/50; here scaled to the
owner-labeled 23-clip session).

Usage: .venv/bin/python scripts/m2_eval.py [diag_dir] [groundtruth_json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    diag_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sessions/m1-diag")
    gt_path = Path(
        sys.argv[2] if len(sys.argv) > 2 else "scripts/groundtruth/session-2026-08-11.json"
    )
    labels = json.loads(gt_path.read_text())["labels"]

    correct = seg_errors = verdict_errors = 0
    rows = []
    for stem, truth in sorted(labels.items()):
        f = diag_dir / f"{stem}.json"
        if not f.exists():
            rows.append((stem, truth, "NOT PROCESSED", "✗"))
            seg_errors += 1
            continue
        d = json.loads(f.read_text())
        events = d["events"]
        if len(events) != 1:
            rows.append((stem, truth, f"{len(events)} events", "✗ seg"))
            seg_errors += 1
            continue
        e = events[0]
        got = e["verdict"]
        ok = got == truth
        correct += ok
        verdict_errors += not ok
        mark = "✓" if ok else "✗"
        rows.append(
            (stem, truth, f"{got}/{e['verdict_confidence']} f{e['frames'][0]}-{e['frames'][1]}", mark)
        )

    for stem, truth, got, mark in rows:
        print(f"{mark:5s} {stem}: truth={truth:4s} engine={got}")
    n = len(labels)
    print(f"\n=== M2 SCORECARD: {correct}/{n} correct "
          f"({seg_errors} segmentation errors, {verdict_errors} verdict errors); "
          f"bar >= {int(n * 0.9 + 0.999)}/{n} (90%) ===")


if __name__ == "__main__":
    main()
