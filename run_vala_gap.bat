@echo off
REM VALA on 09c1414f1b, the one scene it missed. It OOM'd at the default batch_size on this scene --
REM the largest of the twelve at 1.35 M primitives -- so the stochastic Weiszfeld update is chunked
REM smaller. batch_size affects only how many primitives are updated per step, not the estimator,
REM so the result is the same method, just less peak memory.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
cd /d D:\Downloads\baselines\VALA
D:\conda\envs\langsplat\python.exe gaussian_feature_extractor.py -s "D:\Downloads\spp_data_1600\09c1414f1b" -m "Z:\users\rajehyl\powerfoam-archive\refbench_3dgs_12scenes\output\refbench-09c1414f1b" --iteration 30000 --feature_level 0 --use_efficient --batch_size 15000
if errorlevel 1 (echo [vala][FAIL] 09c1414f1b) else (echo [vala][OK] 09c1414f1b)
