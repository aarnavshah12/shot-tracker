"""The upload interface (plan M5, upload half): drag a clip in, get the
annotated mp4 and the stats log back. Local only — models run in-process,
nothing leaves the machine.

Run:  .venv/bin/python -m app.server   (http://127.0.0.1:7878)

One background worker processes jobs sequentially (the models are shared and
not thread-safe); the page polls job status for a live progress bar.
"""

from __future__ import annotations

import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.pipeline import run_upload
from engine.config import EngineConfig

ALLOWED_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
UPLOAD_ROOT = Path("sessions/uploads")

app = FastAPI(title="shot tracker")

_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_models: dict = {}
_worker_started = threading.Lock()
_started = False


def _load_models() -> dict:
    """Real models by default; tests override app.server.MODEL_FACTORY."""
    cfg = EngineConfig.upload()
    from models.detector import RFDETRDetector
    from models.pose import RFDETRPoseModel
    from models.registry import DETECTOR_V0_MODEL_ID

    return {
        "config": cfg,
        "detector": RFDETRDetector(
            DETECTOR_V0_MODEL_ID, cfg.detector.ball_confidence, cfg.detector.rim_confidence
        ),
        "pose_model": RFDETRPoseModel(),
    }


MODEL_FACTORY = _load_models


def _worker() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        try:
            if not _models:
                job["status"] = "loading models"
                _models.update(MODEL_FACTORY())
            job["status"] = "processing"

            def on_progress(done: int, total: int) -> None:
                job["frames_done"], job["frames_total"] = done, total

            result = run_upload(
                job["source"],
                detector=_models["detector"],
                pose_model=_models.get("pose_model"),
                config=_models["config"],
                session_id=f"upload-{job_id}",
                progress=on_progress,
            )
            job["result"] = result
            job["status"] = "done"
        except Exception as exc:  # surfaced to the UI, never a dead job
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"


def _ensure_worker() -> None:
    global _started
    with _worker_started:
        if not _started:
            threading.Thread(target=_worker, daemon=True, name="shot-worker").start()
            _started = True


@app.post("/upload")
async def upload(file: UploadFile) -> dict:
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type {suffix!r}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / f"source{suffix}"
    with open(source, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    _jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "source": str(source),
        "status": "queued",
        "frames_done": 0,
        "frames_total": 0,
    }
    _ensure_worker()
    _queue.put(job_id)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    out = {k: v for k, v in job.items() if k not in ("source",)}
    if job["status"] == "done":
        out["summary"] = job["result"]["summary"]
    return out


def _result_file(job_id: str, key: str) -> Path:
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "job not finished")
    path = Path(job["result"][key] or "")
    if not path.exists():
        raise HTTPException(404, f"{key} not available")
    return path


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "annotated_video"), media_type="video/mp4")


@app.get("/jobs/{job_id}/shots")
def job_shots(job_id: str) -> FileResponse:
    return FileResponse(
        _result_file(job_id, "shots_jsonl"),
        media_type="application/jsonl",
        filename="shots.jsonl",
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>shot tracker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --violet:#7C3AED; --violet-light:#A78BFA; --ink:#F9FAFB; --gray:#9CA3AF;
          --line:#374151; --card:#1F2937; --bg:#111827; --green:#10B981; --red:#EF4444; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink); min-height:100vh;
         font-family:'Inter',system-ui,-apple-system,sans-serif; padding:48px 24px; }
  main { max-width:880px; margin:0 auto; }
  h1 { font-size:28px; font-weight:700; }
  h1 .v { color:var(--violet-light); }
  p.sub { color:var(--gray); font-size:14px; margin-top:6px; }
  #drop { margin-top:32px; border:2px dashed var(--violet); padding:64px 24px;
          text-align:center; cursor:pointer; transition:background .15s; position:relative; }
  #drop:hover, #drop.over { background:#1a1533; }
  #drop .tab { position:absolute; top:-2px; left:-2px; background:var(--violet);
               color:#fff; font:600 12px 'Inter'; padding:3px 8px; }
  #drop b { color:var(--violet-light); }
  #bar { margin-top:24px; height:8px; background:var(--card); display:none; }
  #bar i { display:block; height:100%; width:0; background:var(--violet); transition:width .3s; }
  #status { margin-top:12px; color:var(--gray); font-size:14px; min-height:20px; }
  #result { display:none; margin-top:32px; }
  video { width:100%; border:1px solid var(--line); }
  .stats { display:flex; gap:12px; margin-top:16px; flex-wrap:wrap; }
  .tile { background:var(--card); border:1px solid var(--line); padding:16px 20px; flex:1; min-width:130px; }
  .tile .n { font-size:32px; font-weight:700; }
  .tile .l { font-size:12px; color:var(--gray); text-transform:uppercase; letter-spacing:.08em; margin-top:4px; }
  .tile.make .n { color:var(--green); }
  a.dl { color:var(--violet-light); font-size:14px; display:inline-block; margin-top:16px; margin-right:24px; }
  input[type=file] { display:none; }
</style></head><body><main>
  <h1>shot <span class="v">tracker</span></h1>
  <p class="sub">Drop a clip. Ball, rim, pose, verdicts — processed locally, nothing leaves this machine.</p>
  <div id="drop"><span class="tab">upload 1.00</span>
    <div><b>Drop a video here</b> or click to choose<br>
    <span style="color:var(--gray);font-size:13px">.mov / .mp4 — a driveway shot clip</span></div>
  </div>
  <input type="file" id="file" accept=".mov,.mp4,.m4v,.avi,.mkv">
  <div id="bar"><i></i></div>
  <div id="status"></div>
  <div id="result">
    <video id="video" controls></video>
    <div class="stats" id="tiles"></div>
    <a class="dl" id="dlv">download annotated mp4</a>
    <a class="dl" id="dls">download shots.jsonl</a>
  </div>
<script>
const drop = document.getElementById('drop'), file = document.getElementById('file');
const bar = document.getElementById('bar'), fill = bar.firstElementChild;
const status = document.getElementById('status'), result = document.getElementById('result');
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); go(e.dataTransfer.files[0]); };
file.onchange = () => go(file.files[0]);
async function go(f) {
  if (!f) return;
  result.style.display = 'none'; bar.style.display = 'block'; fill.style.width = '2%';
  status.textContent = 'uploading ' + f.name + '…';
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/upload', { method: 'POST', body: fd });
  if (!r.ok) { status.textContent = 'upload failed: ' + (await r.text()); return; }
  poll((await r.json()).job_id);
}
async function poll(id) {
  const r = await fetch('/jobs/' + id); const j = await r.json();
  if (j.status === 'error') { status.textContent = 'failed: ' + j.error; return; }
  if (j.status !== 'done') {
    const pct = j.frames_total ? Math.round(100 * j.frames_done / j.frames_total) : 2;
    fill.style.width = Math.max(pct, 2) + '%';
    status.textContent = j.status === 'processing'
      ? `processing frame ${j.frames_done} / ${j.frames_total}` : j.status + '…';
    setTimeout(() => poll(id), 700); return;
  }
  fill.style.width = '100%'; status.textContent = 'done';
  const s = j.summary, t = document.getElementById('tiles');
  const fg = s.fg_pct === null ? '—' : s.fg_pct.toFixed(0) + '%';
  t.innerHTML = `
    <div class="tile"><div class="n">${s.attempts}</div><div class="l">attempts</div></div>
    <div class="tile make"><div class="n">${s.makes}</div><div class="l">makes</div></div>
    <div class="tile"><div class="n">${fg}</div><div class="l">fg%</div></div>
    <div class="tile"><div class="n">${s.streaks.best}</div><div class="l">best streak</div></div>`;
  document.getElementById('video').src = '/jobs/' + id + '/video';
  document.getElementById('dlv').href = '/jobs/' + id + '/video';
  document.getElementById('dls').href = '/jobs/' + id + '/shots';
  result.style.display = 'block';
}
</script></main></body></html>"""


if __name__ == "__main__":
    import uvicorn

    print("shot tracker upload interface: http://127.0.0.1:7878")
    uvicorn.run(app, host="127.0.0.1", port=7878, log_level="warning")
