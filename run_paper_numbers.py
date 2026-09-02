"""Every missing paper number, in one resumable overnight run.

USAGE
    python run_paper_numbers.py --smoke          # tiny version of every stage, ~10 min, proves nothing crashes
    python run_paper_numbers.py                  # the real thing
    python run_paper_numbers.py --stages beta,table4_mesh
    python run_paper_numbers.py --list           # what is missing and what each stage fills

RESULTS ARE NEVER RECOMPUTED. Every stage writes `artifacts/overnight/<stage>.json` and, where the
underlying script has its own per-item outputs, those are checked too. A completed stage is skipped
on the next run, so an interrupted night resumes rather than restarts. Nothing is held only in
memory and nothing is printed-but-not-saved.

EVERY SEGMENTATION NUMBER IS PLAIN PER-CELL ARGMAX -- each cell's own feature, cosine against the
raw class names, argmax. No clustering, no codebook, no mode-vote, no diffusion, no reweighting.
That includes the surface metrics: a class's predicted region is the set of points whose OWN cell
argmaxes to that class. Grouping strategies are a separate question and are deliberately not here.

SURFACE METRICS USE THE LABELLED MESH, not its vertices. This is a REDO, not new work: the numbers
currently in Table 4 came from `ablation_surface` (vertex-based), and only `mesh_surface` uses the
mesh. Vertex-based distances have a floor at the vertex spacing (1.26 cm) and the GT->pred direction
falls for free as the predicted set gets denser; the mesh removes both.

WHAT THIS CANNOT FILL, and why -- see `--list` for the live count:
  * Table 3 (tab:lerf) and the LERF rows of tab:recon need 3DGS trained on ramen/teatime/
    waldo_kitchen. Training logs exist for all four but only `figurines` has a checkpoint
    (`recon_lerf_gs/figurines/ckpts/ckpt_29999_rank0.pt`). The other three must be retrained.
  * Table 4's 36 baseline surface cells (LangSplat ... NormLift) need per-point predictions from
    those methods, which means running each repository ourselves. They are published mIoU rows only.
  * tab:adjacency's mIoU column is a DESIGN QUESTION, not a missing run: under plain per-cell argmax
    nothing consumes the graph, so there is no graph-dependent mIoU to report. Either the column is
    dropped (the 23x degree range already makes the point) or a graph-consuming method is
    reintroduced, which contradicts the rest of the table. The PowerFoam mean-degree cell IS
    computable and is filled by the `adjacency` stage.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

OUT = "artifacts/overnight"
PY = r"D:\conda\envs\powerfoam\python.exe"
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]
BETA_DONE = ["scene0000_00", "scene0062_00", "scene0097_00", "scene0347_00"]
BETA_TODO = [s for s in SCENES if s not in BETA_DONE]
ARMS4 = ["pf_tfroz", "pf_nonfroz", "gs_froz", "gs_unfroz"]
LERF = ["figurines", "ramen", "teatime", "waldo_kitchen"]


def path(stage):
    return os.path.join(OUT, f"{stage}.json")


def done(stage):
    p = path(stage)
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("_complete") is True
    except Exception:                                    # noqa: BLE001
        return False


def save(stage, payload, complete=True):
    """Wrap in a dict before stamping. Stages return whatever their underlying script returns --
    run_percell_masked gives a LIST of per-scene rows, others give dicts -- and assuming a dict
    here threw away a completed stage's results at the last step."""
    os.makedirs(OUT, exist_ok=True)
    body = payload if isinstance(payload, dict) else {"rows": payload}
    body["_complete"] = complete
    body["_written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(body, open(path(stage), "w"), indent=1, default=float)
    print(f"  [saved] {path(stage)}", flush=True)


def sh(cmd, tag, timeout=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    dt = (time.time() - t0) / 60
    ok = r.returncode == 0
    print(f"  [{tag}] {'ok' if ok else 'FAILED rc=%d' % r.returncode} ({dt:.1f} min)", flush=True)
    if not ok:
        print("\n".join((r.stdout + r.stderr).splitlines()[-12:]), flush=True)
    return ok, r.stdout


# --------------------------------------------------------------------------- stages
def stage_beta(smoke):
    """tab:matched -- beta median for the 6 scenes still showing \\nm, both arms."""
    scenes = BETA_TODO[:1] if smoke else BETA_TODO
    views = 2 if smoke else 12
    res = {}
    for scene in scenes:
        for arm in ("pf_truefrozen", "gs_froz"):
            key = f"{arm}_{scene}"
            f = f"artifacts/scannet/beta/{key}.json"
            if os.path.exists(f) and not smoke:
                res[key] = json.load(open(f))
                print(f"  [skip] {key} already computed", flush=True)
                continue
            ok, _ = sh([PY, "compute_beta.py", "--scene", scene, "--arm", arm,
                        "--views", str(views)], f"beta {key}")
            if ok and os.path.exists(f):
                res[key] = json.load(open(f))
    return res


def stage_table4_mesh(smoke):
    """Table 4 REDO: plain per-cell argmax, surface metrics against the MESH not the vertices."""
    scenes = ["scene0347_00"] if smoke else SCENES
    arms = ["pf_tfroz"] if smoke else ARMS4
    out = "artifacts/scannet/table4_mesh.json"
    ok, _ = sh([PY, "run_percell_masked.py", "--recons", ",".join(arms),
                "--scenes", ",".join(scenes), "--mults", "2.0", "--surface",
                "--surface-ref", "mesh", "--out", out], "table4-mesh")
    return json.load(open(out)) if ok and os.path.exists(out) else {"error": "see log"}


def stage_spp_surface(smoke):
    """tab:3dseg_spp -- ScanNet++ surface metrics, plain per-cell argmax, against its own mesh."""
    out = "artifacts/scannetpp/spp_surface.json"
    cmd = [PY, "run_spp_surface.py", "--out", out]
    if smoke:
        cmd += ["--scenes", "1"]
    ok, _ = sh(cmd, "spp-surface")
    return json.load(open(out)) if ok and os.path.exists(out) else {"error": "see log"}


def stage_lerf2d(smoke):
    """tab:2dseg -- LERF-OVS 2D relevancy IoU/Acc/locAcc, our row."""
    scenes = LERF[:1] if smoke else LERF
    res = {}
    for s in scenes:
        f = f"artifacts/lerf_ovs/{s}/lerf2d.json"
        if os.path.exists(f) and not smoke:
            res[s] = json.load(open(f))
            continue
        os.makedirs(os.path.dirname(f), exist_ok=True)
        ok, _ = sh([PY, r"D:\Downloads\claude_logs\eval_lerf_iou.py", "--scene", s,
                    "--method", "feature_foam", "--output", f], f"lerf2d {s}")
        if ok and os.path.exists(f):
            res[s] = json.load(open(f))
    return res


def stage_adjacency(smoke):
    """tab:adjacency -- the PowerFoam mean-degree cell (the mIoU column is a design question)."""
    import numpy as np
    import torch
    scenes = SCENES[:1] if smoke else SCENES
    degs = {}
    for s in scenes:
        p = f"output/scannet_{s}_nonfrozen/model.pt"
        if not os.path.exists(p):
            continue
        m = torch.load(p, map_location="cpu", weights_only=False)
        off = m["adjacency_offsets"].numpy()
        d = np.diff(off)
        degs[s] = {"n_prim": int(len(d)), "mean_degree": float(d.mean()),
                   "median_degree": float(np.median(d))}
        print(f"  {s}: mean facet degree {d.mean():.2f}", flush=True)
    if degs:
        degs["_mean_over_scenes"] = float(np.mean([v["mean_degree"] for v in degs.values()
                                                   if isinstance(v, dict)]))
    return degs


STAGES = {
    "beta":        (stage_beta,        "tab:matched, 12 cells -- beta for 6 remaining scenes x 2 arms"),
    "table4_mesh": (stage_table4_mesh, "tab:3dseg REDO -- per-cell argmax + MESH surface, 4 arms x 10 scenes"),
    "spp_surface": (stage_spp_surface, "tab:3dseg_spp, 18 cells -- ScanNet++ surface, per-cell argmax"),
    "lerf2d":      (stage_lerf2d,      "tab:2dseg, 15 cells -- LERF-OVS 2D, our row"),
    "adjacency":   (stage_adjacency,   "tab:adjacency -- PowerFoam mean facet degree"),
}

BLOCKED = [
    ("tab:lerf (10) + tab:recon LERF (4)",
     "3DGS trained on ramen/teatime/waldo_kitchen. Only figurines has a checkpoint."),
    ("tab:3dseg baseline surface (36)",
     "per-point predictions from LangSplat/OpenGaussian/LAGA/THGS/VALA/Occam's/SFS/LUDVIG/"
     "NormLift -- needs each repository run by us; published rows give mIoU only."),
    ("tab:adjacency mIoU column (8)",
     "DESIGN QUESTION: under plain per-cell argmax nothing consumes the graph, so there is no "
     "graph-dependent mIoU. Drop the column, or reintroduce a graph-consuming method."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--smoke", action="store_true", help="tiny version of each stage")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore saved results and recompute")
    a = ap.parse_args()

    if a.list:
        print("RUNNABLE\n")
        for k, (_, why) in STAGES.items():
            print(f"  {k:14s} {'DONE' if done(k) else '    '}  {why}")
        print("\nBLOCKED -- needs something I cannot produce here\n")
        for what, why in BLOCKED:
            print(f"  {what}\n      {why}")
        return

    os.makedirs(OUT, exist_ok=True)
    order = [s.strip() for s in a.stages.split(",") if s.strip() in STAGES]
    print(f"stages: {', '.join(order)}   smoke={a.smoke}\n", flush=True)
    summary = {}
    for st in order:
        if done(st) and not a.force and not a.smoke:
            print(f"=== {st}: already complete, skipping ===", flush=True)
            summary[st] = "skipped (complete)"
            continue
        print(f"=== {st} ===", flush=True)
        t0 = time.time()
        try:
            res = STAGES[st][0](a.smoke)
            save(st if not a.smoke else st + "_smoke", res, complete=not a.smoke)
            summary[st] = f"ok ({(time.time()-t0)/60:.1f} min)"
        except Exception as exc:                          # noqa: BLE001
            print(f"  [{st}] EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            save(st + "_FAILED", {"error": f"{type(exc).__name__}: {exc}"}, complete=False)
            summary[st] = f"FAILED: {type(exc).__name__}"
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:14s} {v}")
    print(f"\nresults in {OUT}/")


if __name__ == "__main__":
    main()
