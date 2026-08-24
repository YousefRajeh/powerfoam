import glob, numpy as np, os
d = "D:/Downloads/powerfoam/artifacts/scannet/scene0347_00/openclip_features_sam"
fs = sorted(glob.glob(os.path.join(d, "*_s.npy")))[:8]
print("n files", len(sorted(glob.glob(os.path.join(d,'*_s.npy')))))
for f in fs:
    s = np.load(f)
    fpath = f.replace("_s.npy","_f.npy")
    feat = np.load(fpath) if os.path.exists(fpath) else None
    line=[os.path.basename(f), str(s.shape), s.dtype.str]
    for L in range(s.shape[0]):
        lv = s[L]
        ids = np.unique(lv[lv>=0])
        areas = [np.sum(lv==i) for i in ids]
        line.append(f"L{L}: min={lv.min()} max={lv.max()} n={len(ids)} medarea={int(np.median(areas)) if areas else 0}")
    if feat is not None: line.append(f"feat={feat.shape}")
    print(" | ".join(line))
