#!/usr/bin/env python3
"""
Extract colored point cloud from COLMAP points3D.txt and save as:
  - ASCII PLY  (readable by PLYLoader in Three.js)
  - JSON metadata
"""

import os, struct, json
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
POINTS3D_TXT = os.path.join(ROOT, "nerf_data", "colmap_text", "points3D.txt")
PLY_OUT      = os.path.join(ROOT, "drone-viewer", "public", "models", "point_cloud.ply")
META_OUT     = os.path.join(ROOT, "drone-viewer", "public", "models", "point_cloud_meta.json")

def read_points3d(path):
    pts, cols = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]),   int(parts[5]),   int(parts[6])
            pts.append((x, y, z))
            cols.append((r, g, b))
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)

def write_ply_binary(path, pts, cols):
    n = len(pts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        data = np.empty(n, dtype=[
            ("x","f4"),("y","f4"),("z","f4"),
            ("red","u1"),("green","u1"),("blue","u1")
        ])
        data["x"]     = pts[:, 0]
        data["y"]     = pts[:, 1]
        data["z"]     = pts[:, 2]
        data["red"]   = cols[:, 0]
        data["green"] = cols[:, 1]
        data["blue"]  = cols[:, 2]
        f.write(data.tobytes())

def main():
    print(f"Reading {POINTS3D_TXT} ...")
    pts, cols = read_points3d(POINTS3D_TXT)
    n = len(pts)
    print(f"  {n:,} points loaded")

    # Compute bounding box for viewer camera positioning
    bbox_min = pts.min(axis=0).tolist()
    bbox_max = pts.max(axis=0).tolist()
    center   = ((pts.min(axis=0) + pts.max(axis=0)) / 2).tolist()
    extent   = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    print(f"Writing {PLY_OUT} ...")
    write_ply_binary(PLY_OUT, pts, cols)
    size_mb = os.path.getsize(PLY_OUT) / 1e6
    print(f"  Done — {size_mb:.1f} MB")

    meta = {
        "num_points": n,
        "bbox_min":   bbox_min,
        "bbox_max":   bbox_max,
        "center":     center,
        "extent":     extent,
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {META_OUT}")
    print(f"\nCenter: {[round(v,2) for v in center]}  Extent: {extent:.2f}")

if __name__ == "__main__":
    main()
