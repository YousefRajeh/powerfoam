@echo off
REM The two scenes each run missed: Occam on 9071e139d9, VALA on 09c1414f1b. Sequential, one after
REM the other, so a single GPU is never shared.
REM
REM Uses the langsplat env, NOT gs-view: gs-view has been restored to splat-distiller's fork-free
REM gsplat 1.5.1 and must stay that way, but Occam and VALA need the 1.4.0 fork that returns
REM "activated"/"significance". langsplat carries that fork plus simple_knn.
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set PY=D:\conda\envs\langsplat\python.exe
set REF=Z:\users\rajehyl\powerfoam-archive\refbench_3dgs_12scenes\output
set DATA=D:\Downloads\spp_data_1600

echo === [occam] 9071e139d9 %TIME% ===
cd /d D:\Downloads\baselines\OccamLGS
if not exist "%REF%\refbench-9071e139d9\point_cloud\iteration_30000\point_cloud.ply" copy "%REF%\refbench-9071e139d9\point_cloud\iteration_30000\scene_point_cloud.ply" "%REF%\refbench-9071e139d9\point_cloud\iteration_30000\point_cloud.ply" >nul
"%PY%" ply_to_chkpt.py --model-path "%REF%\refbench-9071e139d9" --iteration 30000
"%PY%" gaussian_feature_extractor.py -s "%DATA%\9071e139d9" -m "%REF%\refbench-9071e139d9" --iteration 30000 --feature_level 0
if errorlevel 1 (echo [occam][FAIL] 9071e139d9) else (echo [occam][OK] 9071e139d9 %TIME%)

echo === [vala] 09c1414f1b %TIME% ===
cd /d D:\Downloads\baselines\VALA
"%PY%" gaussian_feature_extractor.py -s "%DATA%\09c1414f1b" -m "%REF%\refbench-09c1414f1b" --iteration 30000 --feature_level 0 --use_efficient
if errorlevel 1 (echo [vala][FAIL] 09c1414f1b) else (echo [vala][OK] 09c1414f1b %TIME%)
echo ########## GAP SCENES DONE %TIME% ##########
