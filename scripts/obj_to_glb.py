#!/usr/bin/env python3
"""Convert OBJ mesh to GLB for web deployment."""
import os
import sys

def main():
    try:
        import trimesh
    except ImportError:
        print("Install trimesh: pip install trimesh")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    nerf_data = os.path.join(root, "nerf_data")
    obj_path = os.path.join(nerf_data, "base.obj")
    glb_path = os.path.join(nerf_data, "drone_mesh.glb")

    if not os.path.exists(obj_path):
        print(f"OBJ not found: {obj_path}")
        sys.exit(1)

    print(f"Loading {obj_path}...")
    mesh = trimesh.load(obj_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([m for m in mesh.geometry.values() if isinstance(m, trimesh.Trimesh)])
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh if hasattr(mesh, "export") else list(mesh.geometry.values())[0]

    print(f"Exporting to {glb_path}...")
    mesh.export(glb_path, file_type="glb")
    size_mb = os.path.getsize(glb_path) / (1024 * 1024)
    print(f"Done. drone_mesh.glb = {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
