import hashlib
import json
import time
from pathlib import Path

import torch

manifest = json.loads(open("artifacts/garden/train_views_manifest_all.json").read())
views = manifest["views"]
out_dir = Path("artifacts/garden/timing_openclip_train")
out_dir.mkdir(parents=True, exist_ok=True)

dummy = torch.zeros(420, 648, 512, dtype=torch.float16)

t0 = time.time()
for v in views:
    torch.save(dummy, out_dir / f"probe_{v['id']:06d}.pt")
t_save = time.time() - t0
print(f"torch.save x{len(views)} (dummy 420x648x512 fp16 tensor, ~275KB each): {t_save:.2f}s ({t_save/len(views)*1000:.1f}ms/file)")

t0 = time.time()
for v in views:
    image_bytes = Path(v["image"]).read_bytes()
    hashlib.sha256(image_bytes).hexdigest()
t_hash = time.time() - t0
print(f"read_bytes+sha256 x{len(views)}: {t_hash:.2f}s ({t_hash/len(views)*1000:.1f}ms/file)")

# cleanup probe files
for v in views:
    (out_dir / f"probe_{v['id']:06d}.pt").unlink(missing_ok=True)
