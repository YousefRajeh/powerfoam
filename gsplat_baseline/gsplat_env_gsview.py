r"""Import this before `import gsplat`. See D:\Downloads\powerfoam\gsplat_env_gsview.py's
docstring for the full explanation -- duplicated here so scripts in this
subdirectory don't need sys.path surgery before the very first import."""
import os

_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
_TORCH_LIB = r"C:\Users\rajehyl\AppData\Local\miniconda3\envs\gs-view\Lib\site-packages\torch\lib"

for _dir in (_CUDA_BIN, _TORCH_LIB):
    if os.path.isdir(_dir):
        os.add_dll_directory(_dir)
