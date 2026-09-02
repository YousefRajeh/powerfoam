"""Our vectorised relevancy must match a LITERAL transcription of their get_relevancy, per class.

Their function scores one positive at a time. Ours evaluates all classes at once, which is an
extension -- so the only way to know the extension is faithful is to run their loop verbatim for
each class and compare. A subtle error here (mean instead of worst-case negative, temperature on the
wrong tensor) would produce plausible numbers and silently misrepresent their method.
"""
import sys
sys.path.insert(0,"D:/Downloads/powerfoam")
import torch
torch.use_deterministic_algorithms(True)
from relevancy import relevancy_scores

def their_get_relevancy(embed, pos_embeds, neg_embeds, positive_id):
    """Verbatim transcription of splat-distiller/pre_processing.py::get_relevancy."""
    phrases_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
    p = phrases_embeds.to(embed.dtype)
    output = torch.mm(embed, p.T)
    n_pos = pos_embeds.shape[0]
    positive_vals = output[..., positive_id: positive_id + 1]
    negative_vals = output[..., n_pos:]
    repeated_pos = positive_vals.repeat(1, neg_embeds.shape[0])
    sims = torch.stack((repeated_pos, negative_vals), dim=-1)
    softmax = torch.softmax(10 * sims, dim=-1)
    best_id = softmax[..., 0].argmin(dim=1)
    return torch.gather(softmax, 1,
        best_id[..., None, None].expand(best_id.shape[0], neg_embeds.shape[0], 2))[:, 0, :]

dev="cuda"; P,D,C,N=5000,512,17,4
g=torch.Generator(device=dev).manual_seed(0)
f=torch.nn.functional.normalize(torch.randn(P,D,generator=g,device=dev),dim=-1)
pos=torch.nn.functional.normalize(torch.randn(C,D,generator=g,device=dev),dim=-1)
neg=torch.nn.functional.normalize(torch.randn(N,D,generator=g,device=dev),dim=-1)

ours=relevancy_scores(f,pos,neg)
ref=torch.stack([their_get_relevancy(f,pos,neg,c)[:,0] for c in range(C)],dim=1)
nd=int((ours!=ref).sum()); mx=float((ours-ref).abs().max())
print(f"ours {tuple(ours.shape)} vs their per-class loop {tuple(ref.shape)}")
print(f"  elements differing: {nd:,}/{ours.numel():,}   max abs diff {mx:.3e}")
print(f"  argmax-over-class differing: {int((ours.argmax(1)!=ref.argmax(1)).sum())}")
# Chunking cannot be asserted BITWISE: cuBLAS picks different kernels by matmul shape, so the raw
# f @ pos.T already differs by ~6e-08 between chunk sizes. Measured consequence is what matters --
# the predicted class must not change.
small=relevancy_scores(f,pos,neg,chunk=137)
cd=(small-ours).abs()
flips=int((small.argmax(1)!=ours.argmax(1)).sum())
print(f"  chunk=137 vs 200000: max abs {float(cd.max()):.3e}  argmax flips {flips}/{P}")
ok = nd==0 and flips==0
print("\n"+("PASS - faithful to their implementation" if ok else "FAIL"))
sys.exit(0 if ok else 1)
