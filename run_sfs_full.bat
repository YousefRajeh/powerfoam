@echo off
REM Splat Feature Solver's OWN pipeline, run as they ship it. gsplat JIT-compiles on import and
REM torch shells out to `where cl`, which only resolves after vcvarsall.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\Downloads\splat-distiller
"D:\conda\envs\splat-distiller\python.exe" -u distill.py %*
