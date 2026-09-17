"""Rewrite baseline_eval/manifest.json so its ckpt/features paths resolve on THIS machine.

The manifest stores absolute paths (`D:\\Downloads\\powerfoam\\artifacts\\baseline_eval\\...`)
because it was written on the machine that built the inputs. Copied anywhere else -- another drive,
another user, a share -- every path in it is dead, and the evaluation scripts report `[miss]` for
every tag, which looks like missing data rather than a broken manifest.

This rewrites each entry's `ckpt`/`features` to sit under the directory the manifest itself lives
in, which is the one thing guaranteed to be correct wherever it was copied to. Run it once after
copying. It is idempotent, it VERIFIES every rewritten path exists before saving, and it writes a
timestamped backup first so the original absolute paths are never lost.
"""
import argparse
import json
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="path to the copied manifest.json")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    a = ap.parse_args()

    man_path = os.path.abspath(a.manifest)
    root = os.path.dirname(man_path)                 # the baseline_eval directory itself
    entries = json.load(open(man_path))

    fixed, missing, unchanged = 0, [], 0
    for e in entries:
        for key in ("ckpt", "features"):
            old = e.get(key)
            if not old:
                continue
            # every entry lives at <baseline_eval>/<tag>/<scene>/<file>
            new = os.path.join(root, e["tag"], e["scene"], os.path.basename(old))
            if os.path.normcase(new) == os.path.normcase(old):
                unchanged += 1
                continue
            if not os.path.exists(new):
                missing.append(new)
                continue
            e[key] = new
            fixed += 1

    print(f"manifest : {man_path}")
    print(f"root     : {root}")
    print(f"entries  : {len(entries)}   rewritten: {fixed}   already correct: {unchanged}")
    if missing:
        print(f"\nMISSING {len(missing)} file(s) -- manifest NOT saved. First few:")
        for m in missing[:8]:
            print("  ", m)
        print("\nThe copy is incomplete: rewriting the manifest would point it at files that are\n"
              "not there. Copy the missing tag directories, then re-run.")
        raise SystemExit(1)
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    if fixed:
        bak = f"{man_path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(man_path, bak)
        json.dump(entries, open(man_path, "w"), indent=1)
        print(f"\nbackup   : {bak}\nsaved    : {man_path}")
    else:
        print("\nnothing to do -- every path already resolves here")


if __name__ == "__main__":
    main()
