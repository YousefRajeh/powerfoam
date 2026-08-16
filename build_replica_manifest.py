"""Build the OpenCLIP extraction manifest for room_0's train views.

Manifest "id" here is the TRAIN-SPLIT-LOCAL sequential index (0..786, in
ascending global-frame order), matching accumulate_feature_stats.py's own
convention exactly: it builds data_handler.cameras from ReplicaDataset's
train split (a plain 0-indexed list, no memory of the original global frame
numbers), then does `views_by_id[idx]` for idx in range(len(cameras)) --
i.e. it looks up the manifest by LIST POSITION, not by global frame index.
An earlier version of this file used the global frame index instead, which
looked reasonable but would have silently pulled the wrong image's features
for most views (the exact bug already caught once on the garden pipeline,
where the mismatch ran the other way -- manifest was local, script assumed
global). Filenames (rgb_{global_frame}.png) still reference the real global
frame; only the manifest "id" key is local.
"""
import json

import numpy as np

DATA_DIR = r"D:\Downloads\powerfoam\data\replica\room_0"
OUTPUT_PATH = r"D:\Downloads\powerfoam\artifacts\replica_room0\openclip_train\manifest_input.json"


def build_split(num_images):
    idx = np.arange(num_images)
    test_mask = idx % 8 == 0
    return idx[~test_mask], idx[test_mask]


def main():
    num_frames = 900
    train_idx, test_idx = build_split(num_frames)
    views = [
        {"id": local_id, "image": f"{DATA_DIR}\\rgb\\rgb_{global_frame}.png", "height": 480, "width": 640}
        for local_id, global_frame in enumerate(train_idx.tolist())
    ]
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"views": views}, f, indent=2)
    print(f"wrote {OUTPUT_PATH} with {len(views)} train views "
          f"(local ids 0..{len(views) - 1}, global frames {train_idx.min()}..{train_idx.max()})")


if __name__ == "__main__":
    main()
