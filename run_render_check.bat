@echo off
REM gsplat JIT-compiles on first use and torch shells out to `where cl`, which only resolves after
REM vcvarsall -- same requirement as the gsplat_vis1 builds.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\Downloads\powerfoam
"D:\conda\envs\powerfoam\python.exe" -u render_camera_check.py
