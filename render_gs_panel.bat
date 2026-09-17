@echo off
REM gs-view env: torch 2.7.1+cu118 matches the installed CUDA 11.8, so gsplat's JIT builds and loads.
REM The powerfoam env is torch cu128; forcing CUDA_HOME=11.8 there produced a .pyd that compiled but
REM failed at import with "DLL load failed while importing gsplat_cuda" -- an ABI mismatch, not a
REM compile error. vcvars is still needed because the JIT invokes cl.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set "NVCC_PREPEND_FLAGS=-std=c++17"
cd /d D:\Downloads\powerfoam
"C:\Users\rajehyl\AppData\Local\miniconda3\envs\gs-view\python.exe" render_seg_gsplat_ply.py --ply %1 --cameras %2 --features %3 --class-names "wall,floor,ceiling,sofa,door,kitchen cabinet,bookshelf,cabinet,kitchen counter,refrigerator,doorframe" --view %4 --out %5
