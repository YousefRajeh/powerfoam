import json
import time

import torch
import torch.nn.functional as F
from PIL import Image

from feature_foam_lifting.extract_openclip_features import _patch_features
from feature_foam_lifting.operator import normalize_features

manifest = json.loads(open("artifacts/garden/train_views_manifest_all.json").read())
views = manifest["views"]
device = "cuda"

t0 = time.time()
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
model = model.eval().to(device)
torch.cuda.synchronize()
print(f"model load: {time.time()-t0:.2f}s")

batch_size = 16
t_load, t_infer, t_post = 0.0, 0.0, 0.0
with torch.inference_mode():
    for start in range(0, len(views), batch_size):
        batch_views = views[start:start + batch_size]

        t0 = time.time()
        images = [preprocess(Image.open(v["image"]).convert("RGB")) for v in batch_views]
        imgs = torch.stack(images).to(device)
        torch.cuda.synchronize()
        t_load += time.time() - t0

        t0 = time.time()
        tokens = _patch_features(model, imgs)
        torch.cuda.synchronize()
        t_infer += time.time() - t0

        t0 = time.time()
        for feature, v in zip(tokens, batch_views):
            upsampled = F.interpolate(feature[None].float(), size=(int(v["height"]), int(v["width"])), mode="bilinear", align_corners=False)[0]
            feature_map = normalize_features(upsampled.permute(1, 2, 0)).cpu().to(torch.float16)
        torch.cuda.synchronize()
        t_post += time.time() - t0

print(f"image load+preprocess (CPU decode+resize+stack): {t_load:.2f}s")
print(f"GPU inference (_patch_features): {t_infer:.2f}s")
print(f"postprocess (upsample+normalize+cpu, NOT saved to disk this run): {t_post:.2f}s")
print(f"total (excluding model load): {t_load+t_infer+t_post:.2f}s")
