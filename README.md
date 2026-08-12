# Shot Tracker

Basketball shot tracker that runs from a single phone camera at the edge of a driveway court. Detects ball and rim, tracks each attempt, logs makes/misses, computes arc, entry angle, and release metrics from body pose, and renders an annotated split-screen video.

Two modes, one engine:

- **Upload mode** — drop in a recorded clip, get back an annotated mp4 plus a stats log.
- **Live mode** — the same pipeline consuming a camera feed in real time.

## Layout

```
engine/    ShotEngine, tracker, shot state machine, metrics, calibration
models/    model loading, RF-DETR configs, version pins
render/    renderer, layout, assets
sources/   VideoFileSource, CameraSource
stats/     stats over shots.jsonl, event log
app/       upload interface
tests/     incl. the synthetic-parabola engine test
```

The engine is frame-in/events-out and never knows where frames come from; upload and live mode differ only in source wrapper and model-size config.

## Upload interface

```
.venv/bin/python -m app.server
```

Open http://127.0.0.1:7878, drop a clip, get the annotated mp4 + stats back.
Everything runs locally.

## Development

```
uv venv --python 3.12 .venv
uv pip install -p .venv -e ".[dev]"
.venv/bin/python -m pytest
```
