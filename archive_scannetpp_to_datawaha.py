"""Move artifacts/scannetpp and artifacts/scannetpp_gs to datawaha (Z:), freeing ~335 GB locally.

COPY -> VERIFY -> DELETE, per file, never a bare move. Z: is a network mount; a `move` that fails
part-way leaves the source already unlinked and the destination truncated, and these are the only
copies. This is the same discipline archive_to_datawaha.py uses, and the reason it exists: a
truncated write is exactly how the local disk got into the state that prompted this (a stats file
"finished" in 75 s and then failed to load with `failed finding central directory`).

VERIFICATION. Size equality after copy2, then -- because size alone would not catch a silently
truncated-and-repadded network write -- the first and last 1 MiB of each file are compared byte for
byte. Full hashing of 335 GB over a network mount would dominate the runtime; head+tail catches
truncation and short writes, which are the failure modes actually seen on this mount. Anything that
fails verification is left on BOTH sides and reported, never deleted.

RESUMABLE. A file already present at the destination with a matching size and matching head/tail is
treated as verified and the local copy is removed without re-copying, so an interrupted run can be
re-invoked directly.

--dry-run prints exactly what would move and what would be freed, and touches nothing.
"""
import argparse
import os
import shutil
import stat
import sys

SRC_ROOT = r"D:\Downloads\powerfoam"
DEST_ROOT = r"Z:\users\rajehyl\powerfoam-archive"
TARGETS = ["artifacts/scannetpp", "artifacts/scannetpp_gs"]
EDGE = 1 << 20          # bytes compared at each end


def edges_match(a, b, size):
    n = min(EDGE, size)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        if fa.read(n) != fb.read(n):
            return False
        if size > n:
            fa.seek(size - n)
            fb.seek(size - n)
            if fa.read(n) != fb.read(n):
                return False
    return True


def remove_local(path):
    """Delete the verified-copied source. Git marks pack files (.idx/.pack) read-only, which makes
    os.remove raise PermissionError on Windows -- and an unhandled one aborted a whole 71 GiB run
    part-way through. Clear the read-only bit and retry; if it still refuses, leave the file (it is
    already safely at the destination) and report it rather than killing the run."""
    try:
        os.remove(path)
        return True
    except PermissionError:
        try:
            os.chmod(path, stat.S_IWRITE)
            os.remove(path)
            return True
        except OSError as e:
            print(f"[KEPT-LOCAL] {path}: {e} (copy at destination is verified)", flush=True)
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--src-root", default=SRC_ROOT,
                    help="paths in --targets are relative to this, and the destination mirrors the "
                         "same relative path under the archive root")
    a = ap.parse_args()

    if not os.path.isdir(DEST_ROOT) and not a.dry_run:
        os.makedirs(DEST_ROOT, exist_ok=True)

    moved = skipped = failed = kept_local = 0
    freed = 0
    for target in a.targets:
        src_dir = os.path.join(a.src_root, target.replace("/", os.sep))
        if not os.path.isdir(src_dir):
            print(f"[miss] {target}")
            continue
        for root, _, files in os.walk(src_dir):
            rel_root = os.path.relpath(root, a.src_root)
            dst_root = os.path.join(DEST_ROOT, rel_root)
            for name in files:
                s = os.path.join(root, name)
                d = os.path.join(dst_root, name)
                try:
                    size = os.path.getsize(s)
                except OSError as e:
                    print(f"[FAIL-STAT] {s}: {e}")
                    failed += 1
                    continue

                if a.dry_run:
                    freed += size
                    moved += 1
                    continue

                already = (os.path.exists(d) and os.path.getsize(d) == size
                           and edges_match(s, d, size))
                if not already:
                    os.makedirs(dst_root, exist_ok=True)
                    try:
                        shutil.copy2(s, d)
                    except OSError as e:
                        print(f"[FAIL-COPY] {s}: {e}", flush=True)
                        failed += 1
                        continue
                    if os.path.getsize(d) != size or not edges_match(s, d, size):
                        print(f"[FAIL-VERIFY] {s} -- left on BOTH sides", flush=True)
                        failed += 1
                        continue
                else:
                    skipped += 1
                if not remove_local(s):
                    kept_local += 1
                    continue
                moved += 1
                freed += size
                if moved % 50 == 0:
                    print(f"  {moved} files, {freed / 2**30:.1f} GiB freed", flush=True)

        if not a.dry_run and failed == 0:
            # only prune empty directories; a non-empty one means something failed verification
            for root, dirs, files in os.walk(src_dir, topdown=False):
                if not os.listdir(root):
                    os.rmdir(root)

    verb = "would free" if a.dry_run else "freed"
    print(f"\n{moved} files, {verb} {freed / 2**30:.1f} GiB "
          f"({skipped} already at destination, {kept_local} copied but not deletable, "
          f"{failed} FAILED)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
