# Gaussian Splatting via Nerfstudio

Nerfstudio requires a C++ build (fpsample) which needs **Visual Studio** or **WSL** on Windows.

## Option A: WSL2 (recommended on Windows)

```bash
# In WSL2 Ubuntu
pip install nerfstudio
ns-process-data images --data /mnt/c/Users/Seanm/Desktop/Air\ Link\ imaging/nerfpython/nerf_data/images --output-dir /mnt/c/Users/Seanm/Desktop/Air\ Link\ imaging/nerfpython/nerf_data_processed
ns-train splatfacto --data /mnt/c/Users/Seanm/Desktop/Air\ Link\ imaging/nerfpython/nerf_data_processed
# After training, export for web:
ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir web_export/
```

Copy the exported `.splat` or `.ply` from `web_export/` into the web viewer's `public/` folder.

## Option B: Use the mesh viewer first

The React viewer in `drone-viewer/` already works with `drone_mesh.glb`. You can add Gaussian Splatting later by dropping an exported splat file into `public/` and enabling it in the viewer.
