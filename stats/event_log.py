"""EventLog: appends resolved shots to sessions/<id>/shots.jsonl (plan 7).

File I/O lives here, outside the engine — the engine emits FrameState, this
consumes it (architecture diagram in the plan). Append semantics are the
plan's contract; what is NOT allowed is two processing runs interleaving
colliding shot_ids in one file, so opening a session that already holds shots
raises instead of silently appending or truncating. (Crash-resume for live
mode gets an explicit resume path at M5.)
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.types import FrameState


class EventLog:
    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "shots.jsonl"
        self._fh = None

    def __enter__(self) -> "EventLog":
        if self.path.exists() and self.path.stat().st_size > 0:
            raise FileExistsError(
                f"{self.path} already contains shots from a previous run; "
                "use a fresh session id (shot_ids restart at 1 per run)"
            )
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def consume(self, state: FrameState) -> None:
        self.consume_events(state.events)

    def consume_events(self, events) -> None:
        """Also used for the engine's end-of-stream finalize() events."""
        for event in events:
            self._fh.write(json.dumps(event.to_json_dict()) + "\n")
            self._fh.flush()

    def write_session_metadata(self, metadata: dict) -> None:
        (self.session_dir / "session.json").write_text(json.dumps(metadata, indent=2))
