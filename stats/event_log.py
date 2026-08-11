"""EventLog: writes resolved shots to sessions/<id>/shots.jsonl.

File I/O lives here, outside the engine — the engine emits FrameState, this
consumes it (architecture diagram in the plan). A session dir corresponds to
one processing run: opening truncates any previous shots.jsonl so re-running
a session id can't interleave two runs' shot_ids.
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
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def consume(self, state: FrameState) -> None:
        for event in state.events:
            self._fh.write(json.dumps(event.to_json_dict()) + "\n")
            self._fh.flush()

    def write_session_metadata(self, metadata: dict) -> None:
        (self.session_dir / "session.json").write_text(json.dumps(metadata, indent=2))
