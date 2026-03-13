@echo off
echo ============================================================
echo  Air Link Imaging - Gaussian Splatting Training
echo ============================================================

:: Set up Visual Studio compiler environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

:: Set CUDA arch for RTX 4060
set TORCH_CUDA_ARCH_LIST=8.9
set DISTUTILS_USE_SDK=1

:: Activate conda env
call conda activate gaussian_splat

:: Go to project root
cd /d "C:\Users\Seanm\Desktop\Air Link imaging\nerfpython"

echo.
echo Starting training... This will take 30-60 minutes.
echo Output will be saved to gs_output/ and drone-viewer/public/models/
echo.

python scripts\train_gaussian_splat.py --scene nerf_data --output gs_output --iters 30000

echo.
echo Done! Open http://localhost:3000 to view the Gaussian Splat result.
pause
