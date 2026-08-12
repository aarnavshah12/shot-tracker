# Shot Tracker

Basketball shot tracker that runs from a single phone camera at the edge of a driveway court. Upload a recorded clip and it detects the ball and rim, tracks each attempt, calls makes and misses, computes entry angle and release metrics from body pose, and returns an annotated split-screen video plus a stats log — all locally, nothing leaves your machine.

## Quickstart

```
uv venv --python 3.12 .venv
uv pip install -p .venv -e ".[dev,video,models]"
echo 'ROBOFLOW_API_KEY=...' > .env      # used once, to download the detector weights
.venv/bin/python -m app.server
```

Open http://127.0.0.1:7878, drop a clip, get the annotated mp4 + `shots.jsonl` back.

Best footage: phone on a tripod at the edge of the court, rim in the upper third, full arc and shooter in frame, landscape, 60 fps if available (30 fps works, with less confident verdicts).

## What gets logged per shot

Verdict (make/miss) with a `clean`/`rattled`/`occluded` confidence flag for auditing, entry angle, release velocity and height, arc peak, and elbow/knee/torso angles at release. Session stats: FG%, streaks, make rate by entry-angle bucket.

## Models

- Ball + rim: RF-DETR Small, trained on Roboflow (see `models/registry.py` for the model ID), running locally via `inference` (ONNX).
- Pose: RF-DETR Keypoint, pretrained COCO-17 checkpoint, zero-shot — never trained.

## Layout

```
engine/    ShotEngine, tracker, shot state machine, metrics, calibration
models/    model loading, RF-DETR configs, version pins
render/    split-screen renderer
sources/   VideoFileSource (CameraSource is an unimplemented stub)
stats/     stats over shots.jsonl, event log
app/       upload interface (FastAPI)
tests/     synthetic engine + pipeline tests
```

The engine is frame-in/events-out and never knows where frames come from. A real-time mode is a design goal the architecture leaves room for — a camera source feeding the same engine — but it is **not implemented**: today this processes recorded clips only.

## Development

```
.venv/bin/python -m pytest
```
