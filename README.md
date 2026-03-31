# NeRF Solar Study

3D drone point cloud and mesh viewer with solar simulation. Uses Instant-NGP, COLMAP, and Gaussian splatting for reconstruction.

https://www.youtube.com/watch?v=2FX-flXU8WY
<img width="3832" height="1900" alt="Screenshot 2026-03-30 203039" src="https://github.com/user-attachments/assets/9556515a-0541-4342-b3d6-5956e8bdda33" />


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
