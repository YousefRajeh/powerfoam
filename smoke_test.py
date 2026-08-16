import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())

import gsplat
print("gsplat", gsplat.__version__)

import bsplat
print("bsplat OK", bsplat.__file__)

import gsplat_ext
print("gsplat_ext OK", gsplat_ext.__file__)

import segment_anything_langsplat
print("segment_anything_langsplat OK", segment_anything_langsplat.__file__)

# Exercise an actual CUDA op from gsplat to confirm the built extension loads and runs on-device
device = "cuda"
means = torch.randn(100, 3, device=device)
quats = torch.randn(100, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(100, 3, device=device) * 0.1
opacities = torch.rand(100, device=device)
colors = torch.rand(100, 3, device=device)
viewmat = torch.eye(4, device=device)[None]
K = torch.tensor([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], device=device)[None]
render, alpha, info = gsplat.rasterization(means, quats, scales, opacities, colors, viewmat, K, 64, 64)
print("gsplat.rasterization output shape:", render.shape, "device:", render.device)
print("SMOKE_TEST_PASSED")
