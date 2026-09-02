"""ScanNet++: lift -> solve -> true-facet graph, for all 12 PowerFoam-unfrozen scenes.

Mirrors `run_ogproto_l3_pipeline.py` (which produced the ScanNet `_ogl3` artifacts) so the two
datasets differ ONLY in domain. Every protocol choice is copied verbatim:

  * features `openclip_features_sam_l3`, extracted with LangSplat/OpenGaussian black fill + black
    pad, SAM level 3 only;
  * `--sam-level 0`, because the single-level artifact stores its only row at index 0. Passing 3
    would select an empty slice;
  * geometric-median solve (the project default since the room_0 five-solver comparison).

TWO DELIBERATE DEVIATIONS FROM THE SCANNET DRIVER, both required rather than chosen:

  1. The stats file is KEPT, not deleted. `run_ogproto_l3_pipeline` removes it to save disk, but
     the shipping stack's `mode_vote_refine` reads `AccumulatedFeatureStats.reliability()` from
     exactly that file. Deleting it would make the prerefine stage unrunnable.
  2. `config.yaml` is rewritten IN PLACE (Ibex original preserved as `config_ibex.yaml`). The
     reconstructions were trained on Ibex and their configs carry `data_path: /home/rajehyl/spp_data`;
     `scene:` is a separate field, so only the root changes. It must keep the name `config.yaml`
     because `accumulate_feature_stats_sam.py:135` derives the checkpoint directory by stripping the
     literal "/config.yaml" from the config path -- a renamed copy makes it look for
     `config_local.yaml/model.pt`.

The true-facet adjacency is built here too: it is the graph the diffusion stage runs on, and it was
established as the right choice (+0.80/+1.05/+0.88 over Cech) before any of the current work.
"""
import argparse
import os
import subprocess
import sys
import time

PY = r"D:\conda\envs\powerfoam\python.exe"
DATA = r"D:\Downloads\spp_data_1600"
RECON = r"D:\Downloads\spp_results\full"
FEAT_DIR = "openclip_features_sam_l3"
SUFFIX = "_ogl3"
VARIANT = "nonfrozen"

# Smallest first: the cheap scenes surface a path/protocol error in minutes rather than after the
# 733-image scene has burned an hour.
SPP = ["0d2ee665be", "3864514494", "27dd4da69e", "c50d2d1d42", "578511c8a9", "5942004064",
       "f9f95681fd", "d755b3d9d8", "3db0a1c8f3", "9071e139d9", "e7af285f7d", "09c1414f1b"]


def local_config(scene):
    """Rewrite data_path to the local root; everything else is left exactly as trained."""
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    cfg, bak = os.path.join(ck, "config.yaml"), os.path.join(ck, "config_ibex.yaml")
    if not os.path.exists(cfg):
        return None
    lines = open(cfg).readlines()
    cur = next((l for l in lines if l.startswith("data_path:")), "")
    if cur.strip() == f"data_path: {DATA}":
        return cfg                                   # already localised
    if not os.path.exists(bak):
        with open(bak, "w") as fh:                   # keep the Ibex original once
            fh.writelines(lines)
    with open(cfg, "w") as fh:
        fh.writelines([f"data_path: {DATA}\n" if l.startswith("data_path:") else l
                       for l in lines])
    return cfg


def process(scene):
    art = f"artifacts/scannetpp/{scene}"
    os.makedirs(art, exist_ok=True)
    solved = f"{art}/solved_geometric_median_{VARIANT}{SUFFIX}.pt"
    stats = f"{art}/stats_{VARIANT}{SUFFIX}.pt"
    adjout = f"{art}/adjacency_true_facet.pt"
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")

    feat = os.path.join(DATA, scene, FEAT_DIR)
    n_img = len(os.listdir(os.path.join(DATA, scene, "images")))
    have = len(os.listdir(feat)) if os.path.isdir(feat) else 0
    if have < 2 * n_img:
        print(f"[MISS ] {scene}: features {have}/{2*n_img}", flush=True); return False

    cfg = local_config(scene)
    if cfg is None:
        print(f"[MISS ] {scene}: no config at {ck}", flush=True); return False

    t0 = time.time()
    if not os.path.exists(solved):
        if not os.path.exists(stats):
            print(f"[LIFT ] {scene} ({n_img} views)", flush=True)
            r = subprocess.run(
                [PY, "accumulate_feature_stats_sam.py", "--scene", scene, "--config", cfg,
                 "--feature-folder", feat, "--output", stats, "--sam-level", "0"],
                stdout=open(f"logs_spp_lift_{scene}.log", "w"), stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"[FAIL ] {scene} lift rc={r.returncode} "
                      f"(logs_spp_lift_{scene}.log)", flush=True); return False
        print(f"[SOLVE] {scene}", flush=True)
        r = subprocess.run([PY, "solve_geometric_median.py", "--stats", stats,
                            "--output", solved],
                           stdout=open(f"logs_spp_solve_{scene}.log", "w"),
                           stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[FAIL ] {scene} solve rc={r.returncode}", flush=True); return False
    else:
        print(f"[SKIP ] {scene} already solved", flush=True)

    if not os.path.exists(adjout):
        print(f"[GRAPH] {scene}", flush=True)
        r = subprocess.run([PY, "build_true_facet_graph.py", "--scene", scene,
                            "--variant", VARIANT, "--ckpt-dir", ck, "--output", adjout],
                           stdout=open(f"logs_spp_graph_{scene}.log", "w"),
                           stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[WARN ] {scene} graph rc={r.returncode} "
                  f"(logs_spp_graph_{scene}.log)", flush=True)
    print(f"[OK   ] {scene} {(time.time()-t0)/60:.1f} min", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="")
    a = p.parse_args()
    todo = [x for x in (a.scenes.split(",") if a.scenes else SPP) if x]
    print(f"[plan] {len(todo)} scene(s)", flush=True)
    ok = 0
    for s in todo:
        try:
            ok += bool(process(s))
        except Exception as e:
            print(f"[ERR  ] {s}: {e}", flush=True)
    print(f"[ALL DONE] {ok}/{len(todo)} solved", flush=True)


if __name__ == "__main__":
    main()
