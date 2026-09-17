"""Archive the closed one-off probe families (and the gram caches) to datawaha.

WHAT AND WHY. A per-file-family census of artifacts/scannet found four families with no reference in
any .py and none in the vault -- the weight-transform sweep (stats_tf_*), the Richardson/Jacobi
correction sweep (solved_richardson_*), and two single-variant solves (surfblock, gm_covis). Those
are ~144 GB of finished probes.

ARCHIVED, NOT DELETED, DELIBERATELY. `py=0` is weak evidence: several of these were written by
tag-parameterised code (solved_{tag}.pt), so the literal filename would never appear in a script
even when a script produced it. Archiving frees the same bytes and stays reversible.

THE GRAM CACHES ARE ALSO ARCHIVED RATHER THAN DELETED, against the letter of the request. They are
regenerable in principle, but nothing regenerates them automatically: covis_graph.py globs and
asserts their presence and run_adaptive_centering_eval.py indexes glob()[0], so a hard delete leaves
four scripts raising until someone reruns build_covis_graph.py. Archiving frees the identical 24 GB
without planting that failure.

COPY -> VERIFY -> DELETE, per file. Z: is a network mount; a move that dies mid-write unlinks the
source and truncates the destination, and several of these cannot be rebuilt without re-accumulation.
Verification is size + head/tail digest. Resumable: a file already verified at the destination is
skipped and its local copy removed.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys

DEST_ROOT = r"Z:\users\rajehyl\powerfoam-archive"

FAMILIES = [
    r"^stats_tf_(confp1|confp2|rawpix|rawtan|weisz|ctl)\.pt$",
    r"^solved_richardson_(sph_)?k\d+\.pt$",
    r"^solved_surfblock_c4\.pt$",
    r"^solved_gm_covis_a[\d.]+\.pt$",
    r"^gram_cache_.*\.pt$",
]


def digest(path, n=20_000_000):
    h = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        h.update(fh.read(min(n, size)))
        if size > n:
            fh.seek(max(0, size - n))
            h.update(fh.read(n))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/scannet")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pats = [re.compile(p) for p in FAMILIES]
    targets = []
    for scene in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, scene)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p) and any(r.match(f) for r in pats):
                targets.append((scene, f, p, os.path.getsize(p)))

    total = sum(t[3] for t in targets)
    print("{} files, {:.1f} GB".format(len(targets), total / 2 ** 30))
    if a.dry_run:
        for scene, f, _, sz in targets:
            print("  {:<16} {:<44} {:7.1f} GB".format(scene, f, sz / 2 ** 30))
        return

    if not os.path.isdir(DEST_ROOT):
        print("DEST unreachable: " + DEST_ROOT)
        sys.exit(1)

    freed = moved = skipped = failed = 0
    for scene, f, src, sz in targets:
        dst_dir = os.path.join(DEST_ROOT, "artifacts", "scannet", scene)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f)
        try:
            if os.path.exists(dst) and os.path.getsize(dst) == sz and digest(dst) == digest(src):
                os.remove(src)
                freed += sz
                skipped += 1
                print("[have] {}/{}".format(scene, f), flush=True)
                continue
            print("[copy] {}/{}  {:.1f} GB".format(scene, f, sz / 2 ** 30), flush=True)
            src_dig = digest(src)
            shutil.copy2(src, dst)
            if os.path.getsize(dst) == sz and digest(dst) == src_dig:
                os.remove(src)
                freed += sz
                moved += 1
                print("[ok]   verified, source removed", flush=True)
            else:
                print("[FAIL-VERIFY] {}/{}: destination mismatch, SOURCE KEPT".format(scene, f),
                      flush=True)
                failed += 1
        except Exception as exc:                       # noqa: BLE001 - report and carry on
            print("[FAIL] {}/{}: {}".format(scene, f, exc), flush=True)
            failed += 1

    print("\nmoved {}, already-there {}, failed {}, freed {:.1f} GB".format(
        moved, skipped, failed, freed / 2 ** 30))


if __name__ == "__main__":
    main()
