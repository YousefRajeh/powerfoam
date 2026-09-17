@echo off
REM Build OccamLGS's bundled gsplat into the langsplat env (torch 2.0.0 / cu118 / py3.8) -- the only
REM local env whose toolchain matches the installed CUDA 11.8 and that already has a working
REM simple_knn, which Occam's LGS and VALA both import alongside gsplat.
REM
REM MSVC 14.44's STL hard-fails against CUDA 11.8 with "STL1002: Unexpected compiler version,
REM expected CUDA 12.4 or newer", and every downstream glm static_assert error cascades from it.
REM _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH only moves the problem (already recorded during the
REM radfoam build). The fix is MSVC 14.29, which CUDA 11.8 supports -- and it must come from
REM VS2019's OWN vcvars, since -vcvars_ver only selects toolsets inside the same VS instance.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
REM glm uses the one-argument static_assert, which nvcc only accepts under C++17. The host compiler
REM already gets /std:c++17 from torch, but nvcc does not inherit it -- hence the prepend.
set "NVCC_PREPEND_FLAGS=-std=c++17"
cd /d D:\Downloads\baselines\OccamLGS\submodules\gsplat
rmdir /s /q build 2>nul
D:\conda\envs\langsplat\python.exe -m pip install -e . --no-build-isolation
