@echo off
REM Build LaGa's patched rasterizer (diff_gaussian_rasterization_contrastive_f) into the langsplat
REM env, which already has torch 2.0.0+cu118 and simple_knn and matches the installed CUDA 11.8.
REM
REM Same toolchain that worked for gsplat earlier: VS2019's MSVC 14.29, because 14.44's STL hard
REM fails against CUDA < 12.4 with "STL1002: Unexpected compiler version". nvcc also needs an
REM explicit -std=c++17 -- it does not inherit torch's /std:c++17, and glm's one-argument
REM static_assert is invalid without it.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set "NVCC_PREPEND_FLAGS=-std=c++17"
cd /d D:\Downloads\baselines\LaGa\submodules\diff-gaussian-rasterization_contrastive_f
rmdir /s /q build 2>nul
D:\conda\envs\langsplat\python.exe -m pip install . --no-build-isolation --no-deps
