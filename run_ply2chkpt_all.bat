@echo off
REM Build chkpnt30000.pth for all 12 refbench scenes -- LangSplat's --start_checkpoint.
REM Uses the repo's own GaussianModel.load_ply + capture_rgb, so the tensor layout, activation
REM conventions and SH ordering are theirs. Idempotent: a scene that already has one is skipped.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set PY=D:\conda\envs\langsplat\python.exe
set REF=Z:\users\rajehyl\powerfoam-archive\refbench_3dgs_12scenes\output
cd /d D:\Downloads\baselines\OccamLGS
for %%S in (09c1414f1b 0d2ee665be 27dd4da69e 3864514494 3db0a1c8f3 578511c8a9 5942004064 9071e139d9 c50d2d1d42 d755b3d9d8 e7af285f7d f9f95681fd) do (
  if not exist "%REF%\refbench-%%S\point_cloud\iteration_30000\point_cloud.ply" copy "%REF%\refbench-%%S\point_cloud\iteration_30000\scene_point_cloud.ply" "%REF%\refbench-%%S\point_cloud\iteration_30000\point_cloud.ply" >nul
  "%PY%" ply_to_chkpt.py --model-path "%REF%\refbench-%%S" --iteration 30000
)
echo ########## PLY2CHKPT DONE %TIME% ##########
