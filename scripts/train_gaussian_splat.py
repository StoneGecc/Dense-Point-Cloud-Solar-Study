#!/usr/bin/env python3
"""
Train 3D Gaussian Splatting using gsplat on COLMAP data.

Usage:
    conda run -n gaussian_splat python scripts/train_gaussian_splat.py
    conda run -n gaussian_splat python scripts/train_gaussian_splat.py --iters 10000 --scene nerf_data

This script:
  1. Reads COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
  2. Initializes Gaussians from COLMAP point cloud
  3. Trains using gsplat's rasterization
  4. Exports trained model as PLY + converts to .splat for web viewer
"""

import argparse
import math
import os
import sys
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import gsplat
    from gsplat import rasterization
except ImportError:
    print("gsplat not found. Install with: pip install gsplat")
    sys.exit(1)

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False

ROOT = Path(__file__).parent.parent
WEB_MODELS = ROOT / "drone-viewer" / "public" / "models"

# ── COLMAP reader (manual, no pycolmap dependency) ─────────────────────────────
def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<i", f.read(4))[0]
            model = struct.unpack("<i", f.read(4))[0]
            W = struct.unpack("<Q", f.read(8))[0]
            H = struct.unpack("<Q", f.read(8))[0]
            n_params = [3, 4, 4, 5, 8, 12, 5, 5][model]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cid] = {"model": model, "W": W, "H": H, "params": list(params)}
    return cameras

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts * 24)  # skip 2D observations
            images[iid] = {
                "q": [qw, qx, qy, qz], "t": [tx, ty, tz],
                "camera_id": cid, "name": name.decode()
            }
    return images

def read_points3d_bin(path):
    pts, cols = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)  # pid
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            f.read(8)  # error
            n_obs = struct.unpack("<Q", f.read(8))[0]
            f.read(n_obs * 8)  # track
            pts.append([x, y, z])
            cols.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)

# ── Camera utils ───────────────────────────────────────────────────────────────
def quat_to_rot(q):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

def intrinsics_from_colmap(cam):
    """Return (fx, fy, cx, cy) from COLMAP camera."""
    p = cam["params"]
    W, H = cam["W"], cam["H"]
    model = cam["model"]
    if model == 0:   # SIMPLE_PINHOLE: f, cx, cy
        return p[0], p[0], p[1], p[2]
    elif model == 1: # PINHOLE: fx, fy, cx, cy
        return p[0], p[1], p[2], p[3]
    elif model == 2: # SIMPLE_RADIAL
        return p[0], p[0], p[1], p[2]
    elif model == 3: # RADIAL
        return p[0], p[0], p[1], p[2]
    else:            # opencv-like: fx, fy, cx, cy, ...
        return p[0], p[1], p[2], p[3]

# ── PLY export ─────────────────────────────────────────────────────────────────
def save_ply(path, means, scales, quats, opacities, colors_sh):
    """Save trained Gaussians as PLY (Inria 3DGS format)."""
    N = len(means)
    means_np   = means.detach().cpu().numpy()
    scales_np  = torch.log(scales.detach()).cpu().numpy()  # store log-scale
    quats_np   = quats.detach().cpu().numpy()
    opacities_np = torch.logit(opacities.detach().squeeze(-1)).cpu().numpy()
    sh_np = colors_sh.detach().cpu().numpy().reshape(N, -1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    dtype = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
    ]
    n_sh = sh_np.shape[1]
    for i in range(n_sh):
        dtype.append((f"f_dc_{i}" if i < 3 else f"f_rest_{i-3}", "f4"))
    dtype += [("opacity","f4")]
    dtype += [(f"scale_{i}","f4") for i in range(3)]
    dtype += [(f"rot_{i}","f4") for i in range(4)]

    data = np.zeros(N, dtype=dtype)
    data["x"], data["y"], data["z"] = means_np[:,0], means_np[:,1], means_np[:,2]
    for i in range(n_sh):
        key = f"f_dc_{i}" if i < 3 else f"f_rest_{i-3}"
        data[key] = sh_np[:, i]
    data["opacity"] = opacities_np
    data["scale_0"], data["scale_1"], data["scale_2"] = scales_np[:,0], scales_np[:,1], scales_np[:,2]
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = quats_np[:,0], quats_np[:,1], quats_np[:,2], quats_np[:,3]

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
        )
        for name, fmt in dtype:
            header += f"property float {name}\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    print(f"  Saved PLY: {path} ({N:,} Gaussians)")

# ── Main training loop ─────────────────────────────────────────────────────────
def train(scene_dir, output_dir, n_iters=30000, lr=1e-3, sh_degree=0):
    device = torch.device("cuda")

    sparse_dir = Path(scene_dir) / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = Path(scene_dir) / "colmap_sparse" / "0"
    img_dir = Path(scene_dir) / "images"

    print(f"Reading COLMAP from {sparse_dir} ...")
    cameras = read_cameras_bin(str(sparse_dir / "cameras.bin"))
    images  = read_images_bin(str(sparse_dir / "images.bin"))
    pts3d, cols3d = read_points3d_bin(str(sparse_dir / "points3D.bin"))

    print(f"  {len(cameras)} cameras, {len(images)} images, {len(pts3d):,} points")

    # Build view list (sample up to 100 images for speed)
    view_keys = sorted(images.keys())
    if len(view_keys) > 100:
        step = len(view_keys) // 100
        view_keys = view_keys[::step][:100]

    # Load reference images (resize to 800px max for training speed)
    MAX_PX = 800
    print(f"Loading {len(view_keys)} training images ...")
    view_data = []
    for iid in tqdm(view_keys):
        info = images[iid]
        img_path = img_dir / info["name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        scale = min(MAX_PX / max(W, H), 1.0)
        if scale < 1.0:
            img = img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)

        cam = cameras[info["camera_id"]]
        fx, fy, cx, cy = intrinsics_from_colmap(cam)

        R = quat_to_rot(info["q"])
        t = np.array(info["t"])
        # COLMAP: world→camera; gsplat wants c2w
        R_inv = R.T
        t_inv = -R_inv @ t
        c2w = np.eye(4)
        c2w[:3, :3] = R_inv
        c2w[:3, 3]  = t_inv

        gt = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).to(device)
        view_data.append({
            "gt": gt,
            "c2w": torch.tensor(c2w, dtype=torch.float32, device=device),
            "K": torch.tensor([fx*scale, fy*scale, cx*scale, cy*scale], dtype=torch.float32, device=device),
            "W": img.width, "H": img.height,
        })

    if not view_data:
        print("No images loaded — check image paths in COLMAP")
        sys.exit(1)

    # Initialize Gaussians from COLMAP points
    N = len(pts3d)
    print(f"\nInitializing {N:,} Gaussians ...")
    means    = torch.tensor(pts3d, dtype=torch.float32, device=device, requires_grad=True)
    quats    = torch.zeros(N, 4, device=device);  quats[:, 0] = 1.0
    quats    = torch.nn.Parameter(quats)
    scales   = torch.ones(N, 3, device=device) * 0.01
    scales   = torch.nn.Parameter(scales)
    opacities = torch.ones(N, 1, device=device) * 0.1
    opacities = torch.nn.Parameter(opacities)
    # DC SH colours (3 channels)
    colors_sh = torch.tensor(cols3d, dtype=torch.float32, device=device)
    colors_sh = (colors_sh - 0.5) / 0.28209479   # inverse SH C0
    colors_sh = torch.nn.Parameter(colors_sh.unsqueeze(1))  # (N, 1, 3) SH degree 0

    means_param = torch.nn.Parameter(means.clone())
    optimizer = torch.optim.Adam([
        {"params": [means_param], "lr": lr},
        {"params": [quats],      "lr": lr * 0.1},
        {"params": [scales],     "lr": lr * 0.5},
        {"params": [opacities],  "lr": lr * 0.5},
        {"params": [colors_sh],  "lr": lr * 0.05},
    ], eps=1e-15)

    def rgb_from_sh(sh):
        """SH degree 0 → RGB."""
        return torch.sigmoid(sh.squeeze(1) * 0.28209479 + 0.5)

    print(f"\nTraining {n_iters} iterations ...")
    losses = []
    pbar = tqdm(range(n_iters))
    for step in pbar:
        # Pick random view
        v = view_data[step % len(view_data)]
        H, W = v["H"], v["W"]
        fx, fy, cx, cy = v["K"]

        c2w = v["c2w"].unsqueeze(0)   # (1, 4, 4)
        K_mat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                              device=device, dtype=torch.float32).unsqueeze(0)

        colors = rgb_from_sh(colors_sh)  # (N, 3)
        opac   = torch.sigmoid(opacities)  # (N, 1)
        sc     = torch.exp(scales)          # (N, 3)
        q      = F.normalize(quats, dim=-1) # (N, 4)

        render_out = rasterization(
            means=means_param,
            quats=q,
            scales=sc,
            opacities=opac.squeeze(-1),
            colors=colors,
            viewmats=torch.inverse(c2w),   # world→cam
            Ks=K_mat,
            width=W, height=H,
            near_plane=0.01, far_plane=1000.0,
            sh_degree=None,
            packed=False,
        )
        rendered = render_out[0][0]  # (H, W, 3)

        loss = F.l1_loss(rendered, v["gt"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Densification every 1000 steps (simplified)
        if step % 1000 == 0 and step < n_iters * 0.7:
            with torch.no_grad():
                low_opac = opac.squeeze() < 0.005
                keep = ~low_opac
                means_param.data = means_param.data[keep]
                quats.data        = quats.data[keep]
                scales.data       = scales.data[keep]
                opacities.data    = opacities.data[keep]
                colors_sh.data    = colors_sh.data[keep]
                print(f"  Pruned to {keep.sum():,} Gaussians")
                # Rebuild optimizer states for pruned params
                optimizer.param_groups[0]["params"] = [means_param]
                optimizer.param_groups[1]["params"] = [quats]
                optimizer.param_groups[2]["params"] = [scales]
                optimizer.param_groups[3]["params"] = [opacities]
                optimizer.param_groups[4]["params"] = [colors_sh]

    print(f"\nFinal loss: {losses[-1]:.4f}")

    # Save
    output_dir = Path(output_dir)
    ply_path = output_dir / "point_cloud" / "iteration_final" / "point_cloud.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        save_ply(
            str(ply_path),
            means_param,
            torch.exp(scales),
            F.normalize(quats, dim=-1),
            torch.sigmoid(opacities),
            colors_sh,
        )

    # Copy to web viewer
    import shutil
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(ply_path), str(WEB_MODELS / "gaussian_splat.ply"))
    print(f"  Web copy → {WEB_MODELS / 'gaussian_splat.ply'}")

    # Convert to .splat
    splat_script = ROOT / "scripts" / "ply_to_splat.py"
    import subprocess
    subprocess.run(["python", str(splat_script),
                    str(WEB_MODELS / "gaussian_splat.ply"),
                    str(WEB_MODELS / "scene.splat")])
    print(f"\nTraining complete! Files ready in {WEB_MODELS}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene",  default="nerf_data", help="COLMAP scene directory")
    p.add_argument("--output", default="gs_output",  help="Output directory")
    p.add_argument("--iters",  type=int, default=10000, help="Training iterations (default: 10000)")
    p.add_argument("--lr",     type=float, default=1e-3)
    args = p.parse_args()

    scene_dir  = ROOT / args.scene
    output_dir = ROOT / args.output

    train(str(scene_dir), str(output_dir), n_iters=args.iters, lr=args.lr)
