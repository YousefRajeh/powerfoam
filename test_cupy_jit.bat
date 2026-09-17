@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
"D:\conda\envs\splat-distiller\python.exe" -c "import cupy as cp; x=cp.arange(10,dtype=cp.float32); y=(x*2+1).sum(); cp.cuda.Stream.null.synchronize(); print('cupy JIT OK, sum=',float(y), 'ver', cp.__version__)"
