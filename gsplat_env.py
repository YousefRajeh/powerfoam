"""Import this before `import gsplat` in any script.

gsplat's JIT-compiled CUDA extension links against CUDA 11.8's cudart
(cudart64_110.dll) because that's the only nvcc on this machine, but Python
3.8+ no longer searches PATH for a loaded module's own DLL dependencies --
only directories registered via os.add_dll_directory. torch registers its
own lib/ directory automatically on import, but nothing registers CUDA
11.8's bin/ directory, so without this the extension fails to import with
the uninformative "DLL load failed: The specified module could not be
found" (see the CCCL/MSVC-flags patch in
D:\conda\envs\powerfoam\Lib\site-packages\gsplat\cuda\_backend.py for the two
build-time fixes this pairs with).

Also note: importing gsplat.cuda._backend always calls `where cl` on every
import (not just the first, uncached one), so any script that imports gsplat
must run inside a Visual Studio x64 dev shell -- use run_with_vs.bat
<script.py> [args...] rather than invoking python directly.
"""
import os

_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
_TORCH_LIB = r"D:\conda\envs\powerfoam\Lib\site-packages\torch\lib"

for _dir in (_CUDA_BIN, _TORCH_LIB):
    if os.path.isdir(_dir):
        os.add_dll_directory(_dir)
