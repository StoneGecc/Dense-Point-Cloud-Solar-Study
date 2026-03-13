"""
FastAPI backend for the 3D viewer + processing pipeline.
Run: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue, Empty

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).parent.resolve()
NERF_DATA = ROOT / "nerf_data"
UPLOAD_DIR = NERF_DATA / "uploads"
WEB_MODELS = ROOT / "drone-viewer" / "public" / "models"
SCRIPTS = ROOT / "scripts"
FFMPEG = ROOT / "instant-ngp-bin" / "Instant-NGP-for-RTX-3000-and-4000" / "external" / "ffmpeg" / "ffmpeg-5.1.2-essentials_build" / "bin" / "ffmpeg.exe"
FFPROBE = FFMPEG.parent / "ffprobe.exe"
COLMAP = ROOT / "instant-ngp-bin" / "Instant-NGP-for-RTX-3000-and-4000" / "external" / "colmap" / "COLMAP-3.7-windows-no-cuda" / "COLMAP.bat"

# Global pipeline state
_pipeline_status = {"status": "idle", "last_params": None, "error": None}
_log_listeners = set()
_log_backlog = []
_LOG_BACKLOG_MAX = 500


def _emit_log(line: str):
    _log_backlog.append(line)
    if len(_log_backlog) > _LOG_BACKLOG_MAX:
        _log_backlog.pop(0)
    for q in _log_listeners:
        try:
            q.put_nowait(line)
        except Exception:
            pass


# ── Pipeline runner (blocking, runs in thread) ──────────────────────────────────
def _run_cmd(cmd, cwd=None):
    _emit_log(f"[RUN] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        _emit_log(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Exit code {proc.returncode}")


def _run_pipeline_thread(params: dict):
    global _pipeline_status
    try:
        _pipeline_status["status"] = "running"
        _pipeline_status["error"] = None
        _pipeline_status["last_params"] = params

        raw_path = params.get("video_path", str(NERF_DATA / "DJI_0957.MP4"))
        video_path = (ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        fps = float(params.get("fps", 5))
        max_features = int(params.get("max_features", 4096))
        poisson_depth = int(params.get("poisson_depth", 9))
        min_density = float(params.get("min_density", 0.02))
        smooth_iterations = int(params.get("smooth_iterations", 2))

        images_dir = NERF_DATA / "images"
        sparse_dir = NERF_DATA / "colmap_sparse" / "0"
        colmap_text = NERF_DATA / "colmap_text"
        db_path = NERF_DATA / "colmap.db"

        # 1. Extract frames
        _emit_log("\n[Step 1] Extracting frames...")
        images_dir.mkdir(parents=True, exist_ok=True)
        if not FFMPEG.exists():
            raise FileNotFoundError(f"FFmpeg not found at {FFMPEG}")
        _run_cmd([str(FFMPEG), "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "1", str(images_dir / "%04d.jpg")])
        _emit_log(f"  Extracted frames to {images_dir}")

        # 2. COLMAP
        _emit_log("\n[Step 2] Running COLMAP...")
        if not COLMAP.exists():
            raise FileNotFoundError(f"COLMAP not found at {COLMAP}")
        sparse_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_cmd([str(COLMAP), "feature_extractor",
                  "--database_path", str(db_path),
                  "--image_path", str(images_dir),
                  "--ImageReader.camera_model", params.get("camera_model", "OPENCV"),
                  "--ImageReader.single_camera", "1",
                  "--SiftExtraction.max_num_features", str(max_features)])
        _run_cmd([str(COLMAP), "sequential_matcher", "--database_path", str(db_path)])
        _run_cmd([str(COLMAP), "mapper",
                  "--database_path", str(db_path),
                  "--image_path", str(images_dir),
                  "--output_path", str(sparse_dir.parent)])
        colmap_text.mkdir(parents=True, exist_ok=True)
        _run_cmd([str(COLMAP), "model_converter",
                  "--input_path", str(sparse_dir),
                  "--output_path", str(colmap_text),
                  "--output_type", "TXT"])

        # 3. Export point cloud
        _emit_log("\n[Step 3] Exporting point cloud...")
        _run_cmd([sys.executable, str(SCRIPTS / "extract_point_cloud.py")])

        # 4. Build mesh
        _emit_log("\n[Step 4] Building mesh...")
        _run_cmd([sys.executable, str(SCRIPTS / "build_mesh.py"),
                  "--depth", str(poisson_depth),
                  "--min-density", str(min_density),
                  "--smooth", str(smooth_iterations)])

        _emit_log("\n[Done] Pipeline finished successfully.")
        _pipeline_status["status"] = "done"
    except Exception as e:
        _emit_log(f"\n[ERROR] {e}")
        _pipeline_status["status"] = "error"
        _pipeline_status["error"] = str(e)


def _run_mesh_only_thread(params: dict):
    global _pipeline_status
    try:
        _pipeline_status["status"] = "running"
        _pipeline_status["error"] = None
        poisson_depth = int(params.get("poisson_depth", 9))
        min_density = float(params.get("min_density", 0.02))
        smooth_iterations = int(params.get("smooth_iterations", 2))
        _emit_log("[Rebuild Mesh] Starting...")
        _run_cmd([sys.executable, str(SCRIPTS / "build_mesh.py"),
                  "--depth", str(poisson_depth),
                  "--min-density", str(min_density),
                  "--smooth", str(smooth_iterations)])
        _emit_log("[Done] Mesh rebuild finished.")
        _pipeline_status["status"] = "done"
    except Exception as e:
        _emit_log(f"\n[ERROR] {e}")
        _pipeline_status["status"] = "error"
        _pipeline_status["error"] = str(e)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Air Link Imaging API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _probe_video(video_path: Path) -> dict | None:
    """Run ffprobe to get video metadata. Returns dict with duration_sec, width, height, fps, frame_count_est."""
    if not FFPROBE or not FFPROBE.exists():
        return None
    try:
        result = subprocess.run(
            [
                str(FFPROBE),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        if not streams:
            return None
        s = streams[0]
        width = int(s.get("width", 0))
        height = int(s.get("height", 0))
        duration = float(s.get("duration") or fmt.get("duration", 0) or 0)
        r_frame_rate = s.get("r_frame_rate", "0/1")
        if "/" in r_frame_rate:
            num, den = map(int, r_frame_rate.split("/"))
            fps = num / den if den else 0
        else:
            fps = float(r_frame_rate) if r_frame_rate else 0
        return {
            "duration_sec": round(duration, 2),
            "width": width,
            "height": height,
            "fps_native": round(fps, 2) if fps else None,
            "frame_count_est": int(duration * fps) if fps and duration else None,
        }
    except Exception:
        return None


class PipelineParams(BaseModel):
    video_path: str = ""
    fps: float = 5.0
    max_frames: int = 0
    max_features: int = 4096
    camera_model: str = "OPENCV"
    poisson_depth: int = 9
    min_density: float = 0.02
    smooth_iterations: int = 2
    outlier_std_ratio: float = 2.0


class MeshRebuildParams(BaseModel):
    poisson_depth: int = 9
    min_density: float = 0.02
    smooth_iterations: int = 2


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file. Saves to nerf_data/uploads/ and returns path + metadata."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return {"ok": False, "error": "Invalid or missing video file"}
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / safe_name
    try:
        content = await file.read()
        dest.write_bytes(content)
        size_mb = len(content) / 1e6
        meta = _probe_video(dest)
        rel_path = str(dest.relative_to(ROOT))
        return {
            "ok": True,
            "path": rel_path.replace("\\", "/"),
            "filename": safe_name,
            "size_mb": round(size_mb, 2),
            "metadata": meta,
        }
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return {"ok": False, "error": str(e)}


class ProbeRequest(BaseModel):
    video_path: str = ""


@app.post("/api/video/probe")
def probe_video(req: ProbeRequest):
    """Get metadata for an existing video file (by path relative to project root)."""
    path = Path(req.video_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return {"ok": False, "error": "File not found"}
    meta = _probe_video(path)
    if meta is None:
        return {"ok": False, "error": "Could not probe video"}
    return {"ok": True, "metadata": meta}


@app.get("/api/status")
def get_status():
    return {
        **_pipeline_status,
        "models_info": _get_models_info(),
    }


def _get_models_info():
    info = {"point_cloud": None, "mesh": None, "meta": None}
    meta_path = WEB_MODELS / "point_cloud_meta.json"
    ply_path = WEB_MODELS / "point_cloud.ply"
    glb_path = WEB_MODELS / "drone_mesh.glb"
    if meta_path.exists():
        try:
            info["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if ply_path.exists():
        info["point_cloud"] = {"size_mb": round(ply_path.stat().st_size / 1e6, 2)}
    if glb_path.exists():
        info["mesh"] = {"size_mb": round(glb_path.stat().st_size / 1e6, 2)}
    return info


@app.post("/api/pipeline/run")
def pipeline_run(params: PipelineParams):
    if _pipeline_status["status"] == "running":
        return {"ok": False, "error": "Pipeline already running"}
    p = params.model_dump()
    p["video_path"] = p["video_path"] or str(NERF_DATA / "DJI_0957.MP4")
    thread = threading.Thread(target=_run_pipeline_thread, args=(p,))
    thread.daemon = True
    thread.start()
    return {"ok": True, "message": "Pipeline started"}


@app.post("/api/mesh/rebuild")
def mesh_rebuild(params: MeshRebuildParams):
    if _pipeline_status["status"] == "running":
        return {"ok": False, "error": "Pipeline already running"}
    thread = threading.Thread(target=_run_mesh_only_thread, args=(params.model_dump(),))
    thread.daemon = True
    thread.start()
    return {"ok": True, "message": "Mesh rebuild started"}


@app.get("/api/models/info")
def models_info():
    return _get_models_info()


@app.get("/api/pipeline/log")
async def pipeline_log_stream(request):
    async def event_generator():
        q = Queue()
        _log_listeners.add(q)
        # Send backlog first
        for line in _log_backlog:
            yield {"data": line}
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = q.get(timeout=0.5)
                    yield {"data": line}
                except Empty:
                    yield {"data": ""}
        finally:
            _log_listeners.discard(q)

    return EventSourceResponse(event_generator())


@app.post("/api/status/reset")
def status_reset():
    """Reset status to idle (e.g. after user dismisses done/error)."""
    global _pipeline_status
    _pipeline_status["status"] = "idle"
    _pipeline_status["error"] = None
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
