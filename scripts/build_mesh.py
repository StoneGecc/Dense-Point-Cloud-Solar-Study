#!/usr/bin/env python3
"""
Build a high-quality mesh from COLMAP sparse point cloud using Open3D.

Pipeline:
  1. Read COLMAP points3D.txt (X,Y,Z + RGB)
  2. Estimate normals using camera positions for consistent orientation
  3. Poisson surface reconstruction → watertight mesh
  4. Remove low-density regions (statistical outlier removal)
  5. Export as GLB for the web viewer

Usage:
    python scripts/build_mesh.py
    python scripts/build_mesh.py --depth 10   # higher = finer mesh
"""

import os, sys, argparse
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
POINTS_TXT = ROOT / "nerf_data" / "colmap_text" / "points3D.txt"
IMAGES_TXT = ROOT / "nerf_data" / "colmap_text" / "images.txt"
GLB_OUT    = ROOT / "drone-viewer" / "public" / "models" / "drone_mesh.glb"
PLY_OUT    = ROOT / "drone-viewer" / "public" / "models" / "point_cloud.ply"

try:
    import open3d as o3d
except ImportError:
    print("open3d not installed. Run: pip install open3d")
    sys.exit(1)

# ── read COLMAP points3D.txt ───────────────────────────────────────────────────
def read_points3d(path):
    pts, cols = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            if len(p) < 7: continue
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            cols.append([int(p[4])/255, int(p[5])/255, int(p[6])/255])
    return np.array(pts, np.float32), np.array(cols, np.float32)

# ── read camera centers from images.txt ───────────────────────────────────────
def read_camera_centers(path):
    centers = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 9: i += 1; continue
        qw,qx,qy,qz = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
        tx,ty,tz    = float(parts[5]),float(parts[6]),float(parts[7])
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        t = np.array([tx, ty, tz])
        centers.append(-R.T @ t)   # camera center in world coords
        i += 2  # skip keypoints line
    return np.array(centers, np.float32)

def build_mesh(poisson_depth=9, min_density=0.02, smooth_iterations=2):
    print(f"Reading {POINTS_TXT} ...")
    pts, cols = read_points3d(str(POINTS_TXT))
    print(f"  {len(pts):,} points loaded")

    # ── statistical outlier removal ────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    print("  Removing outliers ...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  {len(pcd.points):,} points after cleaning")

    # ── voxel downsample to even density ──────────────────────────────────────
    # Estimate good voxel size from extent
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(np.array(bbox.max_bound) - np.array(bbox.min_bound))
    voxel = extent / 500   # ~500 voxels across the scene
    print(f"  Voxel downsampling (size={voxel:.3f}) ...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    print(f"  {len(pcd.points):,} points after voxel downsampling")

    # ── normal estimation oriented toward camera positions ─────────────────────
    print("  Estimating normals ...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=30)
    )
    if IMAGES_TXT.exists():
        print("  Orienting normals toward cameras ...")
        centers = read_camera_centers(str(IMAGES_TXT))
        # Use mean camera center as viewpoint for orientation
        vp = centers.mean(axis=0).astype(np.float64)
        pcd.orient_normals_towards_camera_location(vp)
    else:
        pcd.orient_normals_consistent_tangent_plane(30)

    # ── export improved colored point cloud ───────────────────────────────────
    print(f"  Saving point cloud → {PLY_OUT} ...")
    PLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    _write_ply_binary(str(PLY_OUT), pcd)
    print(f"  PLY: {PLY_OUT.stat().st_size/1e6:.1f} MB")

    # ── Poisson surface reconstruction ────────────────────────────────────────
    print(f"  Poisson reconstruction (depth={poisson_depth}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.1, linear_fit=False
    )
    print(f"  Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Remove low-density vertices (artifact trimming)
    densities_np = np.asarray(densities)
    thresh = np.quantile(densities_np, min_density)
    verts_to_remove = densities_np < thresh
    mesh.remove_vertices_by_mask(verts_to_remove)
    print(f"  Trimmed mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Transfer colors from point cloud to mesh
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # fallback
    mesh = _transfer_colors(mesh, pcd)

    # Smooth (configurable)
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
    mesh.compute_vertex_normals()

    # ── export as GLB ─────────────────────────────────────────────────────────
    # Open3D can export .ply; we'll convert to GLB via trimesh
    tmp_ply = str(GLB_OUT).replace(".glb", "_tmp.ply")
    o3d.io.write_triangle_mesh(tmp_ply, mesh)
    print(f"  Intermediate PLY saved, converting to GLB ...")

    try:
        import trimesh
        m = trimesh.load(tmp_ply)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        GLB_OUT.parent.mkdir(parents=True, exist_ok=True)
        m.export(str(GLB_OUT))
        print(f"  GLB saved → {GLB_OUT} ({GLB_OUT.stat().st_size/1e6:.1f} MB)")
        os.remove(tmp_ply)
    except Exception as e:
        print(f"  GLB conversion failed ({e}); keeping PLY at {tmp_ply}")

    print("\nDone!")

def _write_ply_binary(path, pcd):
    pts  = np.asarray(pcd.points,  dtype=np.float32)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    n = len(pts)
    data = np.empty(n, dtype=[
        ("x","f4"),("y","f4"),("z","f4"),
        ("red","u1"),("green","u1"),("blue","u1")
    ])
    data["x"] = pts[:,0]; data["y"] = pts[:,1]; data["z"] = pts[:,2]
    data["red"] = cols[:,0]; data["green"] = cols[:,1]; data["blue"] = cols[:,2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())

def _transfer_colors(mesh, pcd):
    """Transfer nearest-neighbor colors from point cloud to mesh vertices."""
    import open3d as o3d
    mesh_pts = np.asarray(mesh.vertices)
    pcd_pts  = np.asarray(pcd.points)
    pcd_cols = np.asarray(pcd.colors)

    tree = o3d.geometry.KDTreeFlann(pcd)
    colors = np.zeros((len(mesh_pts), 3))
    for i, v in enumerate(mesh_pts):
        _, idx, _ = tree.search_knn_vector_3d(v, 1)
        colors[i] = pcd_cols[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depth",       type=int,   default=9,    help="Poisson depth (8=fast, 10=fine, default=9)")
    p.add_argument("--min-density", type=float, default=0.02, help="Trim low-density fraction (default=0.02)")
    p.add_argument("--smooth",      type=int,   default=2,    help="Smooth iterations (0=off, default=2)")
    args = p.parse_args()
    build_mesh(poisson_depth=args.depth, min_density=args.min_density, smooth_iterations=args.smooth)
