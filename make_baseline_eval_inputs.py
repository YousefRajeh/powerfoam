"""Normalise every baseline's output into the (ckpt, features) pair the existing evals accept.

evaluate_point_cloud_miou.py and eval_semantic_surface_gaussian.py both consume:
    ckpt      -- {"splats": {"means": (N,3), "opacities": (N,) PRE-activation}}   (sigmoid applied
                 by load_gaussian_means_opacities, so raw _opacity is what belongs here)
    features  -- torch tensor (N, 512), one row per Gaussian in `means`

The four baselines store this four different ways:

  LUDVIG    features.npy (P,512) over the FULL frozen Gaussian set; geometry stays in the source
            gsplat checkpoint. No pruning -- N == P.
  Occam     13-element tuple; xyz/opacity/features are all PRUNED consistently (it drops Gaussians
            it never confidently observed), so the tuple is self-contained.
  VALA      same 13-element layout, pruned far harder (its Weiszfeld gate keeps ~12-28%).
  LangSplat 13-element tuple over the FULL set, but features are the autoencoder's 3-d code, not
            CLIP. They must be decoded back to 512-d with THAT SCENE'S decoder before any text
            similarity is meaningful -- a per-scene autoencoder, so the wrong scene's decoder would
            silently produce plausible nonsense.

PRUNING IS RECORDED, NOT RESOLVED. Each baseline keeps a different fraction of the Gaussians, and
how to score a GT point whose only nearby Gaussian was pruned is an open question: counting it wrong
punishes pruning, dropping it rewards pruning. This script therefore emits n_original/n_kept/
kept_frac per scene into a manifest and leaves the decision to the eval, so the choice is visible in
the results rather than baked in here.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

LS = r"D:\Downloads\baselines\LangSplat"
RESULTS = r"D:\Downloads\baselines\results_995"
OCCAM = r"D:\Downloads\baselines\OccamLGS"
RECON = r"D:\Downloads\powerfoam\recon_remote"
# SFS features are produced on 995 (its cuml / pinned-pycolmap chain lives there) and pulled here,
# alongside the gsplat checkpoint they were lifted from, so the pair stays together.
SFS_PULL = r"D:\Downloads\powerfoam\artifacts\sfs_frozen"
NORMLIFT = r"D:\Downloads\NormLift_release"

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]

# baseline -> arm -> (kind, path template). Arm is the reconstruction each paper's own script
# implies: unfrozen for LangSplat/Occam/VALA (plain train.py with densification), frozen for LUDVIG
# (its README follows OpenGaussian Stage 0 exactly).
SOURCES = {
    ("langsplat", "frozen"):   ("langsplat", RESULTS + r"\LangSplat\output\scannet-{scene}_{level}\chkpnt30000.pth"),
    ("langsplat", "unfrozen"): ("langsplat", LS + r"\output_unfroz\scannet-{scene}_{level}\chkpnt30000.pth"),
    ("occam", "unfrozen"):     ("tuple", OCCAM + r"\output_unfroz\scannet-{scene}\chkpnt30000_langfeat_2.pth"),
    ("vala", "frozen"):        ("tuple", RESULTS + r"\VALA\output\scannet-{scene}\none\chkpnt30000_langfeat_0_stochastic_gate.pth"),
    ("vala", "unfrozen"):      ("tuple", RESULTS + r"\VALA\output_unfroz\scannet-{scene}\none\chkpnt30000_langfeat_0_stochastic_gate.pth"),
    ("ludvig", "frozen"):      ("ludvig", RESULTS + r"\ludvig\logs\ScanNet\{scene}\features.npy"),
    # Occam at level 3 (large) alongside its level-2 default: level 2 is only its argparse default
    # and the repo ships no ScanNet script, while level 3 is the level-matched comparison to our
    # own _ogl3 rows. Both are kept.
    ("occam3", "unfrozen"):    ("tuple", OCCAM + r"\output_unfroz\scannet-{scene}\chkpnt30000_langfeat_3.pth"),
    # SFS writes a bare (N,512) tensor next to the gsplat checkpoint rather than a 13-tuple, so its
    # geometry comes from that checkpoint -- same shape as the LUDVIG case.
    ("sfs", "frozen"):         ("sfs", SFS_PULL + r"\{scene}\ckpt_29999_rank0_features.pt"),
    # NormLift: FROZEN per its README ("exactly as in OpenGaussian's Stage 0, fixed positions,
    # densification disabled"), and SAM level 3 per its own hyperparameter table -- so it is
    # the one baseline already level-matched to our _ogl3 rows. Its lift writes a bare
    # (N, 512) tensor, and N equals the scene's GT vertex count exactly (81,369 / 51,610 /
    # ... verified per scene), which confirms the one-Gaussian-per-vertex frozen substitution
    # held; geometry therefore comes from the same gs_froz checkpoint the lift ran against.
    ("normlift", "frozen"):    ("tensor", NORMLIFT + r"\outputs\{scene}\ckpts\point_cloud_features.pt"),
}
SFS_CKPT = SFS_PULL + r"\{scene}\ckpt_29999_rank0.pt"
ARM_CKPT = {"frozen": RECON + r"\gs_froz\{scene}\ckpt.pt",
            "unfrozen": RECON + r"\gs_unfroz\{scene}\ckpt.pt"}


def n_original(arm, scene):
    p = ARM_CKPT[arm].format(scene=scene)
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
    return int(sp["means"].shape[0])


_decoders = {}


def decode_langsplat(feat3, scene, device):
    """3-d autoencoder code -> 512-d CLIP space, using THIS scene's decoder."""
    if scene not in _decoders:
        sys.path.insert(0, os.path.join(LS, "autoencoder"))
        from model import Autoencoder
        ck = os.path.join(LS, "autoencoder", "ckpt", f"scannet-{scene}", "best_ckpt.pth")
        if not os.path.exists(ck):
            raise FileNotFoundError(f"no autoencoder decoder for {scene}: {ck}")
        m = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        m.eval()
        _decoders[scene] = m
    m = _decoders[scene]
    out = []
    with torch.no_grad():
        for i in range(0, feat3.shape[0], 200000):        # chunked: P can be ~3M
            x = feat3[i:i + 200000].to(device).float()
            # NORMALISE FIRST, exactly as LangSplat's render() does:
            #     language_feature_precomp = pc.get_language_feature / (norm + 1e-9)
            # The decoder is only ever trained on unit-norm codes (Autoencoder.forward normalises
            # the encoder output before decoding), but the stored per-Gaussian _language_feature is
            # a free parameter with median norm ~0.07 -- 98% of rows fall below 0.5. Feeding those
            # raw put the decoder far out of distribution and produced 3-7 mIoU, versus ~30 for
            # every other baseline. The +1e-9 keeps never-observed Gaussians (norm exactly 0) at
            # zero instead of NaN, matching render()'s behaviour rather than dropping them.
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
            x = m.decode(x)                                # their own decode(): layers + l2-norm
            out.append(x.cpu())
    return torch.cat(out)


def build(baseline, arm, scene, level, out_root, device):
    kind, tmpl = SOURCES[(baseline, arm)]
    src = tmpl.format(scene=scene, level=level)
    if not os.path.exists(src):
        return None, f"missing {os.path.basename(src)}"

    if kind == "sfs":
        feats = torch.load(src, map_location="cpu", weights_only=False).float()
        ck = torch.load(SFS_CKPT.format(scene=scene), map_location="cpu", weights_only=False)
        sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
        means, opac = sp["means"].float(), sp["opacities"].float()
    elif kind == "tensor":
        feats = torch.load(src, map_location="cpu", weights_only=False).float()
        ck = torch.load(ARM_CKPT[arm].format(scene=scene), map_location="cpu", weights_only=False)
        sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
        means, opac = sp["means"].float(), sp["opacities"].float()
    elif kind == "ludvig":
        feats = torch.from_numpy(np.load(src)).float()
        ck = torch.load(ARM_CKPT[arm].format(scene=scene), map_location="cpu", weights_only=False)
        sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
        means, opac = sp["means"].float(), sp["opacities"].float()
    else:
        mp, _ = torch.load(src, map_location="cpu", weights_only=False)
        means, opac, feats = mp[1].float(), mp[6].float(), mp[7].float()
        if kind == "langsplat":
            feats = decode_langsplat(feats, scene, device)
    if opac.dim() > 1:
        opac = opac.squeeze(-1)

    assert feats.shape[0] == means.shape[0], (feats.shape, means.shape)
    assert feats.shape[1] == 512, feats.shape

    tag = f"{baseline}_{arm}" + (f"_l{level}" if kind == "langsplat" else "")
    d = os.path.join(out_root, tag, scene)
    os.makedirs(d, exist_ok=True)
    torch.save({"splats": {"means": means, "opacities": opac}}, os.path.join(d, "ckpt.pt"))
    torch.save(feats, os.path.join(d, "features.pt"))

    n_all = n_original(arm, scene)
    n_kept = int(means.shape[0])
    return dict(baseline=baseline, arm=arm, scene=scene, level=level, tag=tag,
                n_original=n_all, n_kept=n_kept,
                kept_frac=(n_kept / n_all) if n_all else None,
                ckpt=os.path.join(d, "ckpt.pt"), features=os.path.join(d, "features.pt")), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=r"D:\Downloads\powerfoam\artifacts\baseline_eval")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--only", default=None, help="comma list of baseline_arm tags to build")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    manifest = []
    for (baseline, arm) in SOURCES:
        if a.only and f"{baseline}_{arm}" not in a.only.split(","):
            continue
        levels = [1, 2, 3] if baseline == "langsplat" else [None]
        for scene in a.scenes.split(","):
            for lv in levels:
                rec, msg = build(baseline, arm, scene, lv, a.out_root, a.device)
                if rec is None:
                    print(f"  {baseline:10s} {arm:9s} {scene} L{lv}  SKIP: {msg}", flush=True)
                    continue
                manifest.append(rec)
                print(f"  {rec['tag']:22s} {scene}  kept {rec['n_kept']}/{rec['n_original']}"
                      f" ({100*rec['kept_frac']:.1f}%)", flush=True)
    os.makedirs(a.out_root, exist_ok=True)
    mp = os.path.join(a.out_root, "manifest.json")
    # MERGE, never overwrite. With --only this wrote a manifest containing ONLY the rebuilt
    # tags, silently dropping every other baseline from every downstream evaluator -- it had
    # to be repaired with rebuild_baseline_manifest.py three separate times. Entries are keyed
    # by (tag, scene), so a rebuilt tag replaces its own rows and leaves the rest untouched.
    existing = []
    if os.path.exists(mp):
        try:
            existing = json.load(open(mp))
        except Exception as exc:
            print(f"  existing manifest unreadable ({exc}); writing fresh", flush=True)
    merged = {(r["tag"], r["scene"]): r for r in existing}
    merged.update({(r["tag"], r["scene"]): r for r in manifest})
    out = sorted(merged.values(), key=lambda r: (r["tag"], r["scene"]))
    with open(mp, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n{len(manifest)} built, {len(out)} total entries -> {mp}")


if __name__ == "__main__":
    main()
