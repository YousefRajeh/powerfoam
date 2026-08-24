"""Make PowerFoam's `frozen` config actually freeze the point set.

THE BUG (located by instrumented run, 2026-08-24). `configs/scannet_frozen.yaml` sets
`points_lr_init/final: 0.0` and `densify_from/until: 0`, and those DO work -- an instrumented run
showed bitwise-zero position drift through iteration 850. But `train.py:507` calls

    if i < int(0.95 * args.iterations) and i % 100 == 99:
        num_resampled = model.resample(current_target)
        model.sort_points()

**outside** the densify guard. When the guard is false, `current_target = model.points.shape[0]`,
so `resample()` deletes every cell below a contribution threshold and replaces it with a
duplicate of a surviving cell perturbed by `0.05 * cell_radius` (scene.py:826-834). The COUNT is
preserved exactly -- which is why every count-based check passed and hid the churn for months --
while the point SET is progressively destroyed. `sort_points()` then Morton-permutes, destroying
index identity separately.

Measured dose-response, GT points surviving bitwise:
    400 iters (pre-899): 51610/51610 = 100%
    8,000 iters        : 24655/51610 = 47.8%
    30,000 iters       : 10104/51610 = 19.6%

So every shipped PowerFoam "frozen" ScanNet checkpoint has ~80% of its primitives NOT at GT
points. This is independent of the `points_lr` setting; that fix was necessary but not sufficient.

THE FIX. Add `freeze_points`, and when set, skip the resample/sort/rebuild block entirely. Skipping
`rebuild_adjacency` is correct AND cheaper here: with positions frozen and no resampling, the
triangulation cannot change after the initial build.
"""
import re

TRAIN = "train.py"
CONF = "configs/__init__.py"

# ---- 1. declare the flag -------------------------------------------------------------------
c = open(CONF).read()
if "freeze_points" not in c:
    anchor = "    dry_run: bool = False"
    assert anchor in c, "Params anchor not found"
    c = c.replace(
        anchor,
        "    # When true, the point SET is held fixed: no resampling, no Morton re-sort, no\n"
        "    # adjacency rebuild. Required for OpenGaussian's --frozen_init_pts protocol, where\n"
        "    # primitive i must remain GT point i for the whole run. `points_lr = 0` alone is NOT\n"
        "    # sufficient -- see patch_powerfoam_freeze.py for the measured dose-response.\n"
        "    freeze_points: bool = False\n" + anchor,
        1,
    )
    open(CONF, "w").write(c)
    print("configs/__init__.py: added freeze_points")
else:
    print("configs/__init__.py: freeze_points already present")

# ---- 2. gate the resample block ------------------------------------------------------------
t = open(TRAIN).read()
old = """                if i < int(0.95 * args.iterations) and i % 100 == 99:"""
new = """                if (not args.freeze_points) and i < int(0.95 * args.iterations) and i % 100 == 99:"""
if "not args.freeze_points" not in t:
    assert old in t, "resample guard anchor not found"
    t = t.replace(old, new, 1)
    open(TRAIN, "w").write(t)
    print("train.py: resample/sort_points now gated on freeze_points")
else:
    print("train.py: already gated")

# ---- 3. show the result --------------------------------------------------------------------
t = open(TRAIN).read()
i = t.index("not args.freeze_points")
print("---- context ----")
print(t[max(0, i - 300):i + 220])
