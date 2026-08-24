@echo off
REM Gaussian-splatting UNFROZEN arm, run on the local GPU.
REM
REM Identical to the frozen arm that produced ~/gaussian_baseline_scannet (means_lr 0.0,
REM refine_stop_iter 0) except the two freeze knobs are RELEASED:
REM     means_lr             0.0 -> 1.6e-4   (positions optimise)
REM     strategy.refine-stop-iter 0 -> 15000 (densification runs)
REM Everything else -- init_type sfm, max_steps 30000, data_factor 1 -- is unchanged, so
REM frozen-vs-unfrozen is a clean paired comparison within the Gaussian arm.
REM
REM Must run through cmd with vcvarsall: gsplat JIT-compiles CUDA kernels on Windows and needs
REM cl.exe on PATH. Git bash cannot call vcvarsall.bat.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1

set PY=D:\conda\envs\powerfoam\python.exe
set TRAINER=D:\Downloads\splat-distiller\gaussian_splatting\simple_trainer.py
set PYTHONPATH=D:\Downloads\splat-distiller\submodules\gsplat_ext
set OUTROOT=D:\Downloads\gaussian_unfrozen_scannet
cd /d D:\Downloads\powerfoam

for %%S in (scene0347_00 scene0062_00 scene0097_00 scene0000_00 scene0200_00 scene0070_00 scene0400_00 scene0590_00 scene0645_00 scene0140_00) do (
  if exist "%OUTROOT%\%%S\ckpts\ckpt_29999_rank0.pt" (
    echo [SKIP] %%S already done
  ) else (
    echo [START] %%S %TIME%
    "%PY%" "%TRAINER%" default ^
      --data_dir data\scannet\%%S_colmap ^
      --data_factor 1 ^
      --result_dir "%OUTROOT%\%%S" ^
      --init_type sfm ^
      --max_steps 30000 ^
      --means_lr 1.6e-4 ^
      --strategy.refine-stop-iter 15000 ^
      --eval_steps 1000000000 ^
      --ply_steps 30000 ^
      --disable_viewer --disable_video > "logs_gsunfroz_%%S.log" 2>&1
    echo [DONE ] %%S rc=%ERRORLEVEL% %TIME%
  )
)
echo [ALL DONE] %TIME%
