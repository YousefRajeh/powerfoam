"""Agent-1 audit: numerically compare LangSplat's OpenCLIPNetwork vs splat-distiller's.

Extracts the class source slices from BOTH files, execs them in separate namespaces,
instantiates both, and pushes the SAME image tensor through each encode_image.
Reports cosine + max abs diff of the resulting embeddings.
"""
import sys, types, torch, torchvision, numpy as np
import open_clip
from dataclasses import dataclass, field
from typing import Tuple, Type

LS = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\langsplat\preprocess.py"
SD = r"D:\Downloads\splat-distiller\pre_processing.py"

def load(path, lo, hi, tag):
    src = "".join(open(path, encoding="utf-8").readlines()[lo - 1:hi])
    ns = {"torch": torch, "torchvision": torchvision, "nn": torch.nn,
          "open_clip": open_clip, "dataclass": dataclass, "field": field,
          "Tuple": Tuple, "Type": Type, "np": np}
    exec(compile(src, tag, "exec"), ns)
    return ns

ls = load(LS, 25, 107, "langsplat")
sd = load(SD, 32, 127, "splat_distiller")

print("=== config strings ===")
for n, ns in (("langsplat", ls), ("splat-dist", sd)):
    c = ns["OpenCLIPNetworkConfig"]
    print(f"{n:11s} type={c.clip_model_type!r} pretrained={c.clip_model_pretrained!r} "
          f"dims={c.clip_n_dims} neg={c.negatives} pos={c.positives}")

torch.manual_seed(0)
m_ls = ls["OpenCLIPNetwork"](ls["OpenCLIPNetworkConfig"])
m_sd = sd["OpenCLIPNetwork"](sd["OpenCLIPNetworkConfig"])

print("\n=== transform repr ===")
print("langsplat :", m_ls.process)
print("splat-dist:", m_sd.process)
print("identical transform repr:", repr(m_ls.process) == repr(m_sd.process))

print("\n=== model dtype / device ===")
p1 = next(m_ls.model.parameters()); p2 = next(m_sd.model.parameters())
print("langsplat :", p1.dtype, p1.device)
print("splat-dist:", p2.dtype, p2.device)

# weight equality (catches a different resolved checkpoint)
sd1, sd2 = m_ls.model.state_dict(), m_sd.model.state_dict()
assert sd1.keys() == sd2.keys()
wmax = max((sd1[k].float() - sd2[k].float()).abs().max().item() for k in sd1)
print("max abs weight diff between the two loaded checkpoints:", wmax)

# ---- same input tensor, [0,1] float32 on cuda, mimicking sam_encoder output ----
torch.manual_seed(1234)
x = torch.rand(4, 3, 137, 211, dtype=torch.float32, device="cuda")  # ragged HxW like a crop

pre1 = m_ls.process(x); pre2 = m_sd.process(x)
print("\n=== preprocessed tensor ===")
print("shape", tuple(pre1.shape), "max abs diff:", (pre1 - pre2).abs().max().item())
print("pre1 stats  min/max/mean:", pre1.min().item(), pre1.max().item(), pre1.mean().item())

with torch.no_grad():
    e1 = m_ls.encode_image(x).float()
    e2 = m_sd.encode_image(x).float()

print("\n=== embeddings (raw, as encode_image returns) ===")
print("dtype returned:", m_ls.encode_image(x).dtype, m_sd.encode_image(x).dtype)
print("norms langsplat :", e1.norm(dim=-1).tolist())
print("norms splat-dist:", e2.norm(dim=-1).tolist())
print("L2-normalized inside encode_image? ",
      bool(torch.allclose(e1.norm(dim=-1), torch.ones(e1.shape[0], device=e1.device), atol=1e-2)))
print("MAX ABS DIFF :", (e1 - e2).abs().max().item())
cos = torch.nn.functional.cosine_similarity(e1, e2, dim=-1)
print("COSINE       :", cos.tolist())
print("min cosine   : %.6f" % cos.min().item())

# ---- text embeddings sanity ----
print("\n=== neg_embeds (text) ===")
print("max abs diff:", (m_ls.neg_embeds.float() - m_sd.neg_embeds.float()).abs().max().item())

# ---- what the OFFICIAL open_clip transform would do, for reference ----
_, _, official = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
print("\n=== official open_clip preprocess (reference only) ===")
print(official)
