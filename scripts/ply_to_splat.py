#!/usr/bin/env python3
"""
Convert Gaussian Splatting trained PLY → .splat binary format
for the web viewer (antimatter15/splat, @react-three/drei <Splat>).

The .splat format layout per Gaussian (32 bytes total):
  float32 x, y, z          (12 bytes)
  float32 scale log        ( 4 bytes — packed as single float)
  uint8  r, g, b, opacity  ( 4 bytes)
  uint8[4] rotation quaternion (4 bytes, normalized -128..127 → -1..1)
  uint8[4] scale xyz       (4 bytes, log-scale packed)
  -- totals to 28 bytes in antimatter15 format --

Actual antimatter15 format (62 bytes per splat):
  position: 3×f32  (12 bytes)
  scales:   3×f32  (12 bytes)
  rgba:     4×u8   ( 4 bytes)
  rot:      4×u8   ( 4 bytes)  → quaternion components mapped [0,255]
Total = 32 bytes per splat

Reference: https://github.com/antimatter15/splat
"""

import struct
import sys
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile", "-q"])
    from plyfile import PlyData

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def convert(input_ply: str, output_splat: str):
    print(f"Loading {input_ply} ...")
    ply = PlyData.read(input_ply)
    v = ply["vertex"]

    xyz    = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    # Scales (log-space stored in PLY as scale_0, scale_1, scale_2)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)).astype(np.float32)
    # Opacity (log-odds stored as opacity)
    opacity = sigmoid(np.array(v["opacity"], dtype=np.float32))
    # Rotation quaternion (rot_0..rot_3)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    # Normalize quaternion
    rot_norm = rot / (np.linalg.norm(rot, axis=-1, keepdims=True) + 1e-9)
    # Spherical harmonics DC component → colour (features_dc_0..2)
    sh0 = 0.28209479177387814   # SH C0
    r = np.clip((0.5 + sh0 * np.array(v["f_dc_0"], dtype=np.float32)) * 255, 0, 255)
    g = np.clip((0.5 + sh0 * np.array(v["f_dc_1"], dtype=np.float32)) * 255, 0, 255)
    b = np.clip((0.5 + sh0 * np.array(v["f_dc_2"], dtype=np.float32)) * 255, 0, 255)
    a = np.clip(opacity * 255, 0, 255)

    n = len(xyz)
    print(f"  {n:,} Gaussians")

    # Sort by opacity (front-to-back) for better web rendering
    order = np.argsort(-a)
    xyz    = xyz[order]
    scales = scales[order]
    r, g, b, a = r[order], g[order], b[order], a[order]
    rot_norm = rot_norm[order]

    # Pack into binary buffer: 32 bytes per Gaussian
    # layout: pos(12) + scale(12) + rgba(4) + rot(4)
    buf = bytearray(n * 32)
    pos_arr   = xyz.astype(np.float32)
    scale_arr = scales.astype(np.float32)
    rgba_arr  = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    # rot: map -1..1 → 0..255
    rot_arr = np.clip((rot_norm * 128 + 128), 0, 255).astype(np.uint8)

    view = memoryview(buf)
    pos_bytes   = pos_arr.tobytes()
    scale_bytes = scale_arr.tobytes()
    rgba_bytes  = rgba_arr.tobytes()
    rot_bytes   = rot_arr.tobytes()

    # Interleave manually: for each splat, 12+12+4+4 bytes
    out = np.zeros(n, dtype=[
        ("x","f4"),("y","f4"),("z","f4"),
        ("sx","f4"),("sy","f4"),("sz","f4"),
        ("r","u1"),("g","u1"),("b","u1"),("a","u1"),
        ("q0","u1"),("q1","u1"),("q2","u1"),("q3","u1"),
    ])
    out["x"], out["y"], out["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    out["sx"], out["sy"], out["sz"] = scales[:,0], scales[:,1], scales[:,2]
    out["r"], out["g"], out["b"], out["a"] = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8), a.astype(np.uint8)
    out["q0"], out["q1"] = rot_arr[:,0], rot_arr[:,1]
    out["q2"], out["q3"] = rot_arr[:,2], rot_arr[:,3]

    with open(output_splat, "wb") as f:
        f.write(out.tobytes())

    size_mb = len(out.tobytes()) / 1e6
    print(f"  Saved {output_splat}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ply_to_splat.py input.ply output.splat")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
