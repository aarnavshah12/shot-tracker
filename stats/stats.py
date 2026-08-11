"""Queries over shots.jsonl (plan section 7).

Session summary, streaks, and angle-vs-make-rate splits are all queries over
the event log; nothing here recomputes shot logic. Zone splits (FG% by court
zone) activate at M6 once homography-backed court positions exist.
"""

from __future__ import annotations

import json
from pathlib import Path

ANGLE_BUCKETS = ("<40", "40-43", "43-47", ">47")


def load_shots(shots_jsonl: str | Path) -> list[dict]:
    path = Path(shots_jsonl)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fg_pct(shots: list[dict]) -> float | None:
    if not shots:
        return None
    makes = sum(1 for s in shots if s["verdict"] == "make")
    return 100.0 * makes / len(shots)


def streaks(shots: list[dict]) -> dict:
    """Current and best make streak, in shot order."""
    best = cur = 0
    for s in sorted(shots, key=lambda s: s["shot_id"]):
        cur = cur + 1 if s["verdict"] == "make" else 0
        best = max(best, cur)
    return {"current": cur, "best": best}


def angle_bucket(angle_deg: float) -> str:
    if angle_deg < 40:
        return "<40"
    if angle_deg < 43:
        return "40-43"
    if angle_deg < 47:
        return "43-47"
    return ">47"


def make_rate_by_entry_angle(shots: list[dict]) -> dict[str, dict]:
    """Make rate bucketed by entry angle; shots with a null angle are skipped."""
    buckets = {b: {"attempts": 0, "makes": 0} for b in ANGLE_BUCKETS}
    for s in shots:
        angle = s.get("entry_angle_deg")
        if angle is None:
            continue
        b = buckets[angle_bucket(angle)]
        b["attempts"] += 1
        b["makes"] += s["verdict"] == "make"
    for b in buckets.values():
        b["make_rate"] = (100.0 * b["makes"] / b["attempts"]) if b["attempts"] else None
    return buckets


def session_summary(shots_jsonl: str | Path) -> dict:
    shots = load_shots(shots_jsonl)
    return {
        "attempts": len(shots),
        "makes": sum(1 for s in shots if s["verdict"] == "make"),
        "fg_pct": fg_pct(shots),
        "streaks": streaks(shots),
        "by_entry_angle": make_rate_by_entry_angle(shots),
    }
