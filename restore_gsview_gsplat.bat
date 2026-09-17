@echo off
REM Same toolchain that built the fork: VS2019's 14.29 (14.44's STL rejects CUDA < 12.4) and
REM -std=c++17 for nvcc (glm's one-argument static_assert). Non-editable install so no nested
REM PEP517 step runs without torch on its path.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set "NVCC_PREPEND_FLAGS=-std=c++17"
set PY=C:\Users\rajehyl\AppData\Local\miniconda3\envs\gs-view\python.exe
"%PY%" -m pip uninstall -y gsplat
cd /d D:\Downloads\splat-distiller\submodules\gsplat
rmdir /s /q build 2>nul
"%PY%" -m pip install . --no-build-isolation --no-deps
