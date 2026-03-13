# NeRF Solar Study

3D drone point cloud and mesh viewer with solar simulation. Uses Instant-NGP, COLMAP, and Gaussian splatting for reconstruction.

## Setup

1. **Backend**: `pip install fastapi uvicorn pydantic sse-starlette`
2. **Frontend**: `cd drone-viewer && npm install`
3. **Instant-NGP**: Extract `instant-ngp-rtx4000.zip` to `instant-ngp-bin/` (see api_server.py for expected paths)

## Run

- **API + Viewer**: `start_all.bat`
- **API only**: `uvicorn api_server:app --reload --host 0.0.0.0 --port 8000`
- **Viewer only**: `cd drone-viewer && npm start`

## Data

Place COLMAP outputs and models in `nerf_data/`. The pipeline extracts frames from video, runs COLMAP, and produces point clouds/meshes for the viewer.
