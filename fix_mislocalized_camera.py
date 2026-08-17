"""Re-localize a single mis-registered COLMAP camera via PnP against the existing
reconstruction's 2D-3D correspondences, instead of dropping the image or guessing a pose.

Root cause and full writeup: ResearchVault/Experiments/Experiment-E-lerf-ovs.md
("Related, NOT yet fixed: one severely mislocalized camera in figurines"). figurines'
frame_00162.jpg was registered by COLMAP at [-37.4, -7.6, 145.1], 149.6 units from the
camera centroid, vs. 3.5-6.1 units for every other figurines camera -- almost certainly a
bad feature-match/false loop-closure, not a real pose.

Approach: gather every verified (inlier) SIFT match between frame_00162 and any other
already-registered image, keep only matches whose partner keypoint is already part of a
triangulated 3D point (a real 2D-3D correspondence), then run COLMAP's own robust
PnP+refinement (pycolmap.estimate_and_refine_absolute_pose) on those correspondences using
the SAME camera intrinsics COLMAP already established for this scene. This re-derives the
pose from the same underlying feature evidence, rather than re-guessing or discarding data.

Since this only overwrites one image's extrinsic pose (same image_id, same name, same
count/order of registered images), no downstream camera indexing changes -- existing SAM+CLIP
feature manifests and LERF eval frame lookups (keyed by name) remain valid. Only a retrain is
needed to actually benefit from the corrected pose.

Run in the powerfoam env: D:\\conda\\envs\\powerfoam\\python.exe
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

SCENE_DIR = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs\figurines")
BAD_IMAGE_NAME = "frame_00162.jpg"


def gather_correspondences(db, recon, bad_image_id):
    bad_keypoints = db.read_keypoints(bad_image_id)[:, :2]  # (N, 2) pixel xy

    points2D = []
    points3D = []
    seen_kp_idx = set()

    all_images = db.read_all_images()
    other_ids = [im.image_id for im in all_images if im.image_id != bad_image_id]

    for other_id in other_ids:
        if not db.exists_inlier_matches(bad_image_id, other_id):
            continue
        tvg = db.read_two_view_geometry(bad_image_id, other_id)
        matches = tvg.inlier_matches  # (M, 2); columns follow the (bad_image_id, other_id)
        if matches is None or len(matches) == 0:            # argument order passed above,
            continue                                          # not the canonical pair_id order.
        bad_idx_col, other_idx_col = 0, 1

        if other_id not in recon.images:
            continue
        other_image = recon.images[other_id]

        for m in matches:
            bad_idx = int(m[bad_idx_col])
            other_idx = int(m[other_idx_col])
            if bad_idx in seen_kp_idx:
                continue
            if other_idx >= other_image.num_points2D():
                continue
            pt2D = other_image.points2D[other_idx]
            if not pt2D.has_point3D():
                continue
            xyz = recon.points3D[pt2D.point3D_id].xyz
            points2D.append(bad_keypoints[bad_idx])
            points3D.append(xyz)
            seen_kp_idx.add(bad_idx)

    return np.array(points2D), np.array(points3D)


def main():
    db_path = SCENE_DIR / "distorted" / "database.db"
    recon_path = SCENE_DIR / "distorted" / "sparse" / "0"

    db = pycolmap.Database()
    db.open(str(db_path))
    recon = pycolmap.Reconstruction()
    recon.read(str(recon_path))

    bad_image_id = None
    for im in db.read_all_images():
        if im.name == BAD_IMAGE_NAME:
            bad_image_id = im.image_id
            break
    if bad_image_id is None:
        print(f"ERROR: {BAD_IMAGE_NAME} not found in database")
        sys.exit(1)

    print(f"Found {BAD_IMAGE_NAME} as image_id={bad_image_id}")
    old_center = recon.images[bad_image_id].projection_center()
    print(f"Old (bad) projection center: {old_center}")

    points2D, points3D = gather_correspondences(db, recon, bad_image_id)
    print(f"Gathered {len(points2D)} 2D-3D correspondences from verified matches to other images")

    if len(points2D) < 10:
        print("ERROR: too few correspondences to re-localize reliably, aborting")
        sys.exit(1)

    camera = recon.images[bad_image_id].camera
    result = pycolmap.estimate_and_refine_absolute_pose(points2D, points3D, camera)

    if result is None:
        print("ERROR: pose estimation failed (RANSAC found no consistent pose)")
        sys.exit(1)

    print(f"Estimated pose: num_inliers={result['num_inliers']}/{len(points2D)}")
    new_cam_from_world = result["cam_from_world"]

    other_centers = np.array([
        im.projection_center() for iid, im in recon.images.items() if iid != bad_image_id
    ])
    other_centroid = other_centers.mean(axis=0)
    other_scale = np.median(np.linalg.norm(other_centers - other_centroid, axis=1))

    new_center = new_cam_from_world.inverse().translation
    new_dist = np.linalg.norm(new_center - other_centroid)
    print(f"New projection center: {new_center}, distance from camera centroid: {new_dist:.2f} "
          f"(other cameras' median distance: {other_scale:.2f})")

    if new_dist > 5 * other_scale:
        print("WARNING: new pose is still a spatial outlier relative to the rest of the scene "
              "-- NOT applying automatically, inspect manually.")
        sys.exit(1)

    print("New pose looks consistent with the rest of the scene -- applying.")

    # Apply to the distorted reconstruction (source of truth / for re-undistortion if ever needed).
    bad_image = recon.images[bad_image_id]
    bad_image.frame.set_cam_from_world(bad_image.camera_id, new_cam_from_world)
    backup_distorted = SCENE_DIR / "distorted" / "sparse" / "0_before_camera_fix"
    if not backup_distorted.exists():
        shutil.copytree(recon_path, backup_distorted)
        print(f"Backed up original distorted reconstruction to {backup_distorted}")
    recon.write(str(recon_path))
    print(f"Wrote corrected pose into {recon_path}")

    # Apply the SAME extrinsic pose to the undistorted sparse/0 actually used for training
    # (intrinsics differ there, but image_undistorter does not change world-space camera pose,
    # so the corrected rotation/translation transfers directly).
    train_recon_path = SCENE_DIR / "sparse" / "0"
    backup_train = SCENE_DIR / "sparse" / "0_before_camera_fix"
    train_recon = pycolmap.Reconstruction()
    train_recon.read(str(train_recon_path))

    train_image_id = None
    for iid, im in train_recon.images.items():
        if im.name == BAD_IMAGE_NAME:
            train_image_id = iid
            break
    if train_image_id is None:
        print(f"ERROR: {BAD_IMAGE_NAME} not found in training reconstruction {train_recon_path}")
        sys.exit(1)

    if not backup_train.exists():
        shutil.copytree(train_recon_path, backup_train)
        print(f"Backed up original training reconstruction to {backup_train}")

    train_image = train_recon.images[train_image_id]
    train_image.frame.set_cam_from_world(train_image.camera_id, new_cam_from_world)
    train_recon.write(str(train_recon_path))
    print(f"Wrote corrected pose into {train_recon_path} (this is what train.py actually reads)")

    db.close()
    print("Done. A figurines retrain is needed to benefit from the corrected pose.")


if __name__ == "__main__":
    main()
