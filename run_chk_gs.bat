@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
cd /d D:\Downloads\powerfoam
"D:\conda\envs\powerfoam\python.exe" -u chk_gs.py
