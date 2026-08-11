"""Import this before `import gsplat` in any gs-view-env script.

See gsplat_env.py's docstring for the full explanation -- same fix, pointed
at the gs-view conda env's torch/lib instead of powerfoam's. Every script
that imports gsplat must still run inside a Visual Studio x64 dev shell
(gsplat.cuda._backend always calls `where cl` on import) -- use
run_with_vs_gsview.bat <script.py> [args...].
"""
import os

_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
_TORCH_LIB = r"C:\Users\rajehyl\AppData\Local\miniconda3\envs\gs-view\Lib\site-packages\torch\lib"

for _dir in (_CUDA_BIN, _TORCH_LIB):
    if os.path.isdir(_dir):
        os.add_dll_directory(_dir)
