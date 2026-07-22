"""Run persistence + background job runner.

Every run lives in runs/<run_id>/ with:
  run.json          – metadata (name, mode, language, timestamps)
  segments.json     – transcript + translations (the 'memory' of the run)
  job_status.json   – live status of background jobs
  result_<kind>.json– job results
  media + outputs

Background jobs run in daemon threads writing status/results to DISK, so a
finished job survives page refreshes & reconnects (within the app process).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path("runs")
_THREADS: dict[tuple[str, str], threading.Thread] = {}


def _run_dir(run_id: str) -> Path:
    return ROOT / run_id


def new_run() -> str:
    run_id = "run_" + time.strftime("%Y%m%d_%H%M%S")
    d = _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({
        "id": run_id, "created": time.time(),
        "label": run_id, "mode": "", "target": "",
    }), encoding="utf-8")
    return run_id


def run_dir(run_id: str) -> Path:
    d = _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ metadata
def save_meta(run_id: str, **fields) -> None:
    p = _run_dir(run_id) / "run.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta.update(fields)
    meta["updated"] = time.time()
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def list_runs() -> list[dict]:
    out = []
    if ROOT.exists():
        for d in ROOT.iterdir():
            p = d / "run.json"
            if d.is_dir() and p.exists():
                try:
                    out.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def delete_run(run_id: str) -> None:
    shutil.rmtree(_run_dir(run_id), ignore_errors=True)


# ------------------------------------------------------------------ segments
def save_segments(run_id: str, segments: list[dict]) -> None:
    (_run_dir(run_id) / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_segments(run_id: str) -> list[dict] | None:
    p = _run_dir(run_id) / "segments.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ small json
def save_json(run_id: str, name: str, data) -> None:
    """Persist a small JSON-able object for the run (e.g. song ranges)."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    (_run_dir(run_id) / f"{safe}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_json(run_id: str, name: str, default=None):
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    p = _run_dir(run_id) / f"{safe}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def artifact(run_id: str, name: str) -> str | None:
    p = _run_dir(run_id) / name
    return str(p) if p.exists() else None


# ------------------------------------------------------------------ flags
def set_flag(run_id: str, name: str, value) -> None:
    """Tiny JSON flag file used for job<->UI communication (ask/decide etc.)."""
    p = _run_dir(run_id) / f"flag_{name}.json"
    if value is None:
        p.unlink(missing_ok=True)
        return
    try:
        p.write_text(json.dumps({"v": value}), encoding="utf-8")
    except Exception:
        pass


def get_flag(run_id: str, name: str, default=None):
    p = _run_dir(run_id) / f"flag_{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("v", default)
        except Exception:
            pass
    return default


# ------------------------------------------------------------------ jobs
def _status_path(run_id: str) -> Path:
    return _run_dir(run_id) / "job_status.json"


def _write_status(run_id: str, kind: str, **fields) -> None:
    p = _status_path(run_id)
    data: dict[str, Any] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault(kind, {})
    data[kind].update(fields)
    data[kind]["ts"] = time.time()
    try:
        p.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def status(run_id: str, kind: str) -> dict:
    p = _status_path(run_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get(kind, {})
        except Exception:
            pass
    return {}


def job_running(run_id: str, kind: str) -> bool:
    t = _THREADS.get((run_id, kind))
    if t is not None:
        return t.is_alive()
    return status(run_id, kind).get("status") == "running"


def job_start(run_id: str, kind: str, fn: Callable[[Callable], Any]) -> None:
    """Run fn(progress_cb) in a background thread; status+result go to disk."""

    def _cb(pct: float, msg: str):
        _write_status(run_id, kind, status="running", pct=float(pct), msg=msg)

    # write the initial status SYNCHRONOUSLY (prevents UI poll race)
    _write_status(run_id, kind, status="running", pct=0.0, msg="starting…")

    def _wrap():
        try:
            result = fn(_cb)
            (_run_dir(run_id) / f"result_{kind}.json").write_text(
                json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8"
            )
            _write_status(run_id, kind, status="done", pct=1.0, msg="done")
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=3)
            _write_status(run_id, kind, status="error", msg=str(exc), tb=tb)

    t = threading.Thread(target=_wrap, daemon=True, name=f"job-{run_id}-{kind}")
    _THREADS[(run_id, kind)] = t
    t.start()


def job_result(run_id: str, kind: str) -> Any:
    p = _run_dir(run_id) / f"result_{kind}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
