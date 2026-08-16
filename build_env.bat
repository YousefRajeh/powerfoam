@echo on
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (echo FAILED_VCVARSALL & exit /b 1)

set "CUDA_HOME=D:\conda\envs\splat-distiller\Library"
set "CUDA_PATH=D:\conda\envs\splat-distiller\Library"
set "LIB=D:\conda\envs\splat-distiller\Library\lib;%LIB%"
set "DISTUTILS_USE_SDK=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=D:\conda\envs\splat-distiller;D:\conda\envs\splat-distiller\Scripts;D:\conda\envs\splat-distiller\Library\bin;C:\Program Files\Git\cmd;C:\Program Files\Git\mingw64\bin;%PATH%"

where cl.exe
where nvcc.exe
where python.exe

cd /d "%~1"
if errorlevel 1 (echo FAILED_CD & exit /b 1)

echo RUN_CWD=%CD%

%~2
if errorlevel 1 (echo FAILED_COMMAND & exit /b 1)

echo BUILD_SUCCESS
