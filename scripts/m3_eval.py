"""M3 acceptance: elbow/knee angles logged on >= 80% of shots, nulls (not
garbage) on the rest. Reads the diagnostics dumps produced with pose enabled.

Usage: .venv/bin/python scripts/m3_eval.py [diag_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    diag_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sessions/m1-diag")
    shots = with_form = 0
    pose_frames = total_frames = 0
    for f in sorted(diag_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if d["rows"] and "pose" not in d["rows"][0]:
            print(f"FAIL: {f.name} predates the pose-enabled diagnostics — "
                  "delete sessions/m1-diag and re-run scripts/m1_diagnostics.py")
            raise SystemExit(2)
        total_frames += d["frames"]
        pose_frames += sum(1 for r in d["rows"] if r.get("pose"))
        for e in d["events"]:
            shots += 1
            has = e["elbow_deg"] is not None and e["knee_deg"] is not None
            with_form += has
            print(f"{d['clip']}: {e['verdict']}/{e['verdict_confidence']} "
                  f"elbow={e['elbow_deg']} knee={e['knee_deg']} "
                  f"tilt={e['shoulder_hip_deg']} rel_h={e['release_height_m']} "
                  f"rel_v={e['release_velocity_ms']} {'✓' if has else '∅ (nulled)'}")
    if not shots:
        print("FAIL: no shots found in diagnostics — nothing to evaluate")
        raise SystemExit(2)
    pct = 100 * with_form / shots
    print(f"\npose coverage: {pose_frames}/{total_frames} frames "
          f"({100 * pose_frames / max(1, total_frames):.0f}%)")
    print(f"=== M3 ACCEPTANCE: elbow+knee on {with_form}/{shots} shots "
          f"({pct:.0f}%)  [target >= 80%] ===")
    raise SystemExit(0 if pct >= 80 else 1)


if __name__ == "__main__":
    main()
