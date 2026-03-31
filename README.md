# Dense Point Cloud Solar Study

Turn aerial footage into a 3D model for solar exposure analysis.
<img width="3826" height="1887" alt="Screenshot 2026-03-30 202722" src="https://github.com/user-attachments/assets/605db801-cd97-4c9a-9a2e-11e74cbddcc5" />

This project uses drone photogrammetry and site-specific solar simulation to create a 3D model of existing conditions and measure how sunlight hits that geometry over time.

Traditional solar studies often rely on CAD models, which can miss real-world surroundings and unplanned building modifications. Drone footage can be used as an up to date source of truth for existing conditions. This project turns that footage into a 3D model that can be processed for solar analysis. The results can be explored interactively and exported as an OBJ for use in other tools.

## Features

- Interactive 3D viewer for point clouds and meshes
<img width="2268" height="1892" alt="Screenshot 2026-03-30 210932" src="https://github.com/user-attachments/assets/f95637c2-fe8c-4bd3-86be-152478eb597b" />


- Solar simulation by date, time, and location
<img width="2276" height="1894" alt="Screenshot 2026-03-30 211046" src="https://github.com/user-attachments/assets/85a43ff4-2e70-4177-a191-a01a0b4cbadc" />


- Exposure heatmap with adjustable settings
<img width="2273" height="1903" alt="Screenshot 2026-03-30 211212" src="https://github.com/user-attachments/assets/3c5966bb-2ab0-4585-99f8-5ad8ca65f992" />



- OBJ/MTL export with baked color results

- FastAPI pipeline for frame extraction, COLMAP, and mesh generation





## Tech stack

React · Three.js / React Three Fiber · SunCalc · FastAPI · COLMAP / Instant-NGP–oriented tooling (see setup below)

---

## Quick start

### Prerequisites

- **Python 3** with FastAPI stack for the API  
- **Node.js** for the viewer  
- **Instant-NGP** (optional, for full NeRF-style path)—extract per your GPU build into `instant-ngp-bin/` as expected by `api_server.py`

### Install

1. **Backend:** `pip install fastapi uvicorn pydantic sse-starlette`  
2. **Frontend:** `cd drone-viewer && npm install`  
3. **Instant-NGP:** extract `instant-ngp-rtx4000.zip` (or matching build) to `instant-ngp-bin/`

### Run

| Mode | Command |
|------|---------|
| **API + viewer (Windows)** | `start_all.bat` |
| **API only** | `uvicorn api_server:app --reload --host 0.0.0.0 --port 8000` |
| **Viewer only** | `cd drone-viewer && npm start` |

### Data

Place COLMAP outputs and pipeline assets under `nerf_data/`. The pipeline can extract frames from video, run COLMAP, and feed point clouds and meshes into the viewer.
