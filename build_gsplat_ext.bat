@echo off
REM Complete gsplat's link ONCE so ninja stops retrying it on every import.
REM
REM WHY: gsplat's load_extension() calls torch's _jit_compile, catches the torch>=2.7 TypeError and
REM RETRIES -- that retry runs ninja. A link that once failed with LNK1104 (which happens whenever
REM another process has gsplat_cuda.pyd mapped) stays dirty in .ninja_log, so every later import
REM retries the link and fails again while any gsplat job runs. Self-perpetuating; cost four runs.
REM
REM Do NOT move TORCH_EXTENSIONS_DIR: gsplat cannot actually rebuild from scratch against this torch
REM (the retry path errors and falls back to importing a pre-existing library), so the default cache
REM and its working .pyd are the only thing that loads. Run this with NO gsplat process running.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
cd /d "C:\Users\rajehyl\AppData\Local\torch_extensions\torch_extensions\Cache\py311_cu128\gsplat_cuda"
"D:\conda\envs\powerfoam\python.exe" -m ninja
