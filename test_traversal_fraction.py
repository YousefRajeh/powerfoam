"""Does the fraction of a cell a ray actually crosses predict how good that cell's feature is?

MOTIVATION. `A_rj = alpha*T` with `alpha = 1 - exp(-tau)` and `tau = sigma_j * dt_rj`. Measured on
scene0347_00, `alpha/tau` has median 0.926 and 87.2% of cells sit at tau < 0.5, so we are in the
near-linear regime and the weight already IS the density integral along the traversed segment.

What the weight does NOT distinguish is HOW MUCH OF THE CELL that segment covers: a ray clipping
1% of a large cell and a ray crossing 100% of a small one get the same weight when their tau
matches. Recovering `dt = tau/sigma` on real data, the typical ray crosses only **7.8%** of its
cell (92.8% of cells under 25%, 60.2% under 10%), and rho(radius, fraction) = -0.134, so big cells
are systematically the least fully crossed.

That is only harmless if a cell is semantically homogeneous -- exactly the assumption most likely
to fail on large cells. This tests it directly: if the traversed fraction carries real signal,
cells whose rays only clip them should be misclassified more often.

BASELINE TO BEAT. The per-cell ray mass `(A^T 1)_j` was the best predictor found in the 14-paper
coverage campaign (Spearman -0.74 to -0.85 vs per-cell error, beating Tuy, the null-space shuttle,
the visual hull and the photo hull). A new statistic is only interesting if it adds signal ON TOP
of that, so the partial correlation controlling for ray mass is the number that matters, not the
raw one.

CPU-only, ~2-5 min: everything expensive is already cached in the gram cache.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

SCENE = "scene0347_00"
GT = rf"D:\Downloads\scannet_pointcept\train\{SCENE}"
CACHE = f"artifacts/scannet/{SCENE}/gram_cache_K6_l3_v54.pt"
SOLVED = f"artifacts/scannet/{SCENE}/solved_geometric_median_nonfrozen.pt"
CKPT = f"output/scannet_{SCENE}_nonfrozen/model.pt"


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out, via ranks + residuals."""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    rz = (rz - rz.mean()) / rz.std()
    ex = rx - (rx @ rz) / len(rz) * rz
    ey = ry - (ry @ rz) / len(rz) * rz
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    P = int(c["P"]); sup = c["support"].double().numpy()
    keys = c["S_keys"].numpy(); vals = c["S_vals"].double().numpy()
    j, l = keys // P, keys % P
    diag = np.zeros(P); dm = (j == l); diag[j[dm]] = vals[dm]

    m = torch.load(CKPT, map_location="cpu", weights_only=False)
    centers = np.asarray(m["points"].float().cpu()).astype(np.float64)
    rad = np.asarray(m["radii"].float().cpu()).reshape(-1).astype(np.float64)
    sig = F.softplus(m["density"].float().cpu(), beta=100).numpy().reshape(-1).astype(np.float64)

    # typical per-ray weight for each cell, and the optical depth / segment it implies
    alpha = np.zeros(P); ok = sup > 0
    alpha[ok] = diag[ok] / sup[ok]
    tau = -np.log(np.clip(1 - alpha, 1e-12, None))
    dt = np.where(sig > 1e-9, tau / np.maximum(sig, 1e-9), 0.0)
    frac = dt / np.maximum(2 * rad, 1e-12)          # <- the quantity under test

    # per-cell truth: majority GT label of the points falling in that cell
    coord = np.load(f"{GT}/coord.npy").astype(np.float64)
    seg = np.load(f"{GT}/segment20.npy").astype(np.int64)
    print(f"assigning {len(coord)} GT points to {P} cells ...", flush=True)
    # nearest-centre is the Voronoi answer; the power correction uses radii, so search a
    # candidate set and pick the true power-cell argmin  ||x-c||^2 - r^2.
    d, idx = cKDTree(centers).query(coord, k=32, workers=-1)
    powd = d ** 2 - (rad[idx] ** 2)
    owner = idx[np.arange(len(coord)), powd.argmin(axis=1)]

    solved = torch.load(SOLVED, map_location="cpu", weights_only=True)
    feats = solved["primitive_features"].float()
    valid = solved["valid_mask"].numpy()

    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           SCANNET20_CLASS_NAMES)
    names = [n for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n in SCANNET20_CLASS_NAMES]
    tf = embed_class_names(names, "cpu")
    ids = [SCANNET20_CLASS_NAMES.index(n) for n in names]

    unit = F.normalize(feats, dim=-1)
    pred = (unit @ F.normalize(tf, dim=-1).T.cpu()).argmax(-1).numpy()

    # a cell is scorable if it owns >=1 GT point whose label is in our class set
    keep = np.isin(seg[np.arange(len(seg))], ids)
    cell_lab = -np.ones(P, dtype=np.int64)
    for cid in np.unique(owner[keep]):
        lbls = seg[keep][owner[keep] == cid]
        if len(lbls):
            v, ct = np.unique(lbls, return_counts=True)
            cell_lab[cid] = v[ct.argmax()]
    scor = (cell_lab >= 0) & valid & (sup > 0) & (frac > 0)
    err = (np.array([ids[p] for p in pred])[scor] != cell_lab[scor]).astype(float)

    fr, rm, rr = frac[scor], sup[scor], rad[scor]
    print(f"\nscorable cells: {scor.sum()}   misclassified: {err.mean()*100:.2f}%")
    print("\nSpearman rho vs per-cell error (more negative = better predictor):")
    print(f"  ray mass (A^T 1)_j   [BASELINE]  {spearmanr(rm, err)[0]:+.4f}")
    print(f"  traversed fraction dt/diameter   {spearmanr(fr, err)[0]:+.4f}")
    print(f"  cell radius                      {spearmanr(rr, err)[0]:+.4f}")
    print(f"  ray mass / radius                {spearmanr(rm/rr, err)[0]:+.4f}")
    print("\nPARTIAL rho, controlling for ray mass (does it ADD anything?):")
    print(f"  traversed fraction | ray mass    {partial_spearman(fr, err, rm):+.4f}")
    print(f"  cell radius        | ray mass    {partial_spearman(rr, err, rm):+.4f}")

    print("\nmisclassification rate by traversed-fraction quintile:")
    qs = np.quantile(fr, [0, .2, .4, .6, .8, 1.0])
    for a in range(5):
        s = (fr >= qs[a]) & (fr <= qs[a + 1])
        print(f"  frac {qs[a]:.3f}-{qs[a+1]:.3f}: err {err[s].mean()*100:5.2f}%  n={s.sum()}")


if __name__ == "__main__":
    main()
