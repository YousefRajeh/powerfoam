@echo on
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (echo FAILED_VCVARSALL & exit /b 1)

set "PATH=D:\conda\envs\splat-distiller\Library\bin;%PATH%"
set "CUDA_HOME=D:\conda\envs\splat-distiller\Library"
set "CUDA_PATH=D:\conda\envs\splat-distiller\Library"
set "LIB=D:\conda\envs\splat-distiller\Library\lib;%LIB%"
set "DISTUTILS_USE_SDK=1"

where cl.exe
where nvcc.exe

call C:\Users\rajehyl\AppData\Local\miniconda3\condabin\conda.bat activate splat-distiller
if errorlevel 1 (echo FAILED_CONDA_ACTIVATE & exit /b 1)

echo AFTER_ACTIVATE_CD=%CD%
where cl.exe
where nvcc.exe
where python.exe

cd /d "C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\a26251ff-e9aa-482b-a969-e8560ac3f508\scratchpad\splat-distiller\submodules\gsplat"
if errorlevel 1 (echo FAILED_CD & exit /b 1)

echo BUILD_CWD=%CD%

pip install --no-build-isolation -e .
if errorlevel 1 (echo FAILED_PIP_INSTALL & exit /b 1)

echo BUILD_SUCCESS
