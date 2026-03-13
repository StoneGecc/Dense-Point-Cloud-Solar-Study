#!/usr/bin/env python3
"""
Full pipeline: Video → COLMAP → Gaussian Splatting → Web export
Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4 --output output_name

Steps:
  1. Extract frames from video (FFmpeg)
  2. Run COLMAP (feature extraction, matching, sparse reconstruction)
  3. Train 3D Gaussian Splatting
  4. Export .ply (colored point cloud for immediate web viewing)
  5. Convert trained splat to .splat format for web viewer
"""

import argparse
import os
import sys
import shutil
import subprocess
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
GS_DIR   = ROOT / "gaussian-splatting"
FFMPEG   = ROOT / "instant-ngp-bin" / "Instant-NGP-for-RTX-3000-and-4000" / "external" / "ffmpeg" / "ffmpeg-5.1.2-essentials_build" / "bin" / "ffmpeg.exe"
COLMAP   = ROOT / "instant-ngp-bin" / "Instant-NGP-for-RTX-3000-and-4000" / "external" / "colmap" / "COLMAP-3.7-windows-no-cuda" / "COLMAP.bat"
WEB_OUT  = ROOT / "drone-viewer" / "public" / "models"

def run(cmd, cwd=None, env=None):
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode}")
        sys.exit(result.returncode)

def step_extract_frames(video_path: Path, out_dir: Path, fps: float = 2.0):
    """Extract frames using FFmpeg."""
    if not FFMPEG.exists():
        print(f"FFmpeg not found at {FFMPEG}. Please install FFmpeg.")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    run([str(FFMPEG), "-y", "-i", str(video_path),
         "-vf", f"fps={fps}", "-q:v", "1",
         str(out_dir / "%04d.jpg")])
    imgs = list(out_dir.glob("*.jpg"))
    print(f"  Extracted {len(imgs)} frames")
    return imgs

def step_colmap(image_dir: Path, sparse_dir: Path, db_path: Path):
    """Run COLMAP feature extraction, matching, and sparse reconstruction."""
    if not COLMAP.exists():
        print(f"COLMAP not found at {COLMAP}. Please install COLMAP.")
        sys.exit(1)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run([str(COLMAP), "feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(image_dir),
         "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1",
         "--SiftExtraction.use_gpu", "1"])

    run([str(COLMAP), "sequential_matcher",
         "--database_path", str(db_path)])

    run([str(COLMAP), "mapper",
         "--database_path", str(db_path),
         "--image_path", str(image_dir),
         "--output_path", str(sparse_dir)])

def step_train_gaussian_splatting(data_dir: Path, output_dir: Path):
    """Train Gaussian Splatting using the gaussian-splatting repo."""
    if not GS_DIR.exists():
        print(f"Gaussian Splatting repo not found at {GS_DIR}")
        print("Please run: git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting")
        sys.exit(1)

    train_script = GS_DIR / "train.py"
    # Activate conda env and run training
    run(["conda", "run", "-n", "gaussian_splat", "python", str(train_script),
         "-s", str(data_dir),
         "--model_path", str(output_dir),
         "--iterations", "30000"])

def step_export_splat(model_dir: Path, out_web: Path):
    """
    Convert trained Gaussian Splatting PLY to .splat format for the web viewer.
    The trained model is in model_dir/point_cloud/iteration_30000/point_cloud.ply
    """
    ply_path = model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if not ply_path.exists():
        # Try latest iteration
        pc_dir = model_dir / "point_cloud"
        if pc_dir.exists():
            iters = sorted(pc_dir.iterdir())
            if iters:
                ply_path = iters[-1] / "point_cloud.ply"

    if not ply_path.exists():
        print(f"Trained PLY not found at {ply_path}")
        return

    out_web.mkdir(parents=True, exist_ok=True)
    # Copy PLY to web dir for inspection
    shutil.copy(str(ply_path), str(out_web / "gaussian_splat.ply"))
    print(f"  Gaussian splat PLY → {out_web / 'gaussian_splat.ply'}")

    # Convert to .splat binary format (antimatter15 format for drei <Splat>)
    convert_script = ROOT / "scripts" / "ply_to_splat.py"
    if convert_script.exists():
        run(["python", str(convert_script),
             str(ply_path), str(out_web / "scene.splat")])
        print(f"  Web splat → {out_web / 'scene.splat'}")

def step_export_colored_cloud(colmap_text_dir: Path, out_web: Path):
    """Extract colored point cloud from COLMAP and save as PLY for web viewing."""
    pts_txt = colmap_text_dir / "points3D.txt"
    if not pts_txt.exists():
        print(f"points3D.txt not found; skipping colored cloud export")
        return
    extract_script = ROOT / "scripts" / "extract_point_cloud.py"
    if extract_script.exists():
        run(["python", str(extract_script)])
    print(f"  Colored point cloud exported to {out_web / 'point_cloud.ply'}")

# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Video → COLMAP → 3DGS → Web")
    p.add_argument("--video", required=True, help="Input video file")
    p.add_argument("--output", default="scene", help="Output name (used for folder)")
    p.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate (default: 2)")
    p.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP if already run")
    p.add_argument("--skip-train",  action="store_true", help="Skip 3DGS training")
    args = p.parse_args()

    video  = Path(args.video).resolve()
    scene  = ROOT / "scenes" / args.output
    images = scene / "images"
    sparse = scene / "sparse"
    db     = scene / "colmap.db"
    colmap_text = scene / "colmap_text"
    gs_out = scene / "gaussian_splat_output"

    print(f"\n{'='*60}")
    print(f"  Air Link Imaging — 3D Reconstruction Pipeline")
    print(f"{'='*60}")
    print(f"  Input : {video}")
    print(f"  Output: {scene}")

    # 1. Extract frames
    print("\n[Step 1] Extracting frames...")
    step_extract_frames(video, images, fps=args.fps)

    # 2. COLMAP
    if not args.skip_colmap:
        print("\n[Step 2] Running COLMAP...")
        step_colmap(images, sparse, db)
        # Export text format
        colmap_text.mkdir(exist_ok=True)
        run([str(COLMAP), "model_converter",
             "--input_path",  str(sparse / "0"),
             "--output_path", str(colmap_text),
             "--output_type", "TXT"])
    else:
        print("[Step 2] Skipping COLMAP (--skip-colmap)")

    # 3. Export colored cloud for immediate preview
    print("\n[Step 3] Exporting colored point cloud for web preview...")
    step_export_colored_cloud(colmap_text, WEB_OUT)

    # 4. Train Gaussian Splatting
    if not args.skip_train:
        print("\n[Step 4] Training Gaussian Splatting (this takes ~30 min)...")
        # GS expects images/ and sparse/ in same dir
        step_train_gaussian_splatting(scene, gs_out)
    else:
        print("[Step 4] Skipping 3DGS training (--skip-train)")

    # 5. Export .splat for web
    print("\n[Step 5] Exporting splat for web viewer...")
    step_export_splat(gs_out, WEB_OUT)

    print(f"\n{'='*60}")
    print(f"  Done! Files in {WEB_OUT}")
    print(f"  Run: cd drone-viewer && npm start")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
