@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\Downloads\powerfoam
"D:\conda\envs\splat-distiller\python.exe" -u verify_weights_match_vis.py %*
