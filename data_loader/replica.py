import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from powerfoam.camera import TorchCamera
from .metric3D import Metric3DEstimator

# Habitat-Sim's default pinhole sensor (hfov=90 deg), the convention used by
# every paper that renders this exact pre-rendered Replica sequence
# (semantic-nerf, NICE-SLAM, vMAP): fx = fy = width / 2, cx/cy at the pixel
# center. NICE-SLAM's own Replica config confirms this at a different
# resolution (fx=600 for W=1200, i.e. fx/W=0.5) -- same ratio applied here.
NATIVE_WIDTH = 640
NATIVE_HEIGHT = 480


class ReplicaDataset(Dataset):
    """Pre-rendered Replica sequence (semantic-nerf/NICE-SLAM/vMAP format):
    a flat rgb/ folder, a single traj_w_c.txt (one flattened 4x4
    camera-to-world matrix per line, OpenCV convention, already the
    convention PowerFoam/TorchCamera expects -- no blender2opencv-style
    conversion needed), and fixed fov-90 intrinsics. No COLMAP/SfM points
    exist for this data, same as BlenderDataset's synthetic case.

    Held-out split follows this project's own convention elsewhere
    (sorted/natural-order index, test = idx % 8 == 0), not a
    dataset-provided train/test file, since the raw download only ships one
    undivided trajectory per sequence.
    """

    def __init__(self, datadir, split, downsample, alpha_format_on_disk, use_metric3d):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.alpha_format_on_disk = alpha_format_on_disk
        self.use_metric3d = use_metric3d

        rgb_dir = os.path.join(datadir, "rgb")
        traj_path = os.path.join(datadir, "traj_w_c.txt")
        num_frames = len([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(".png")])

        poses_all = np.loadtxt(traj_path).reshape(-1, 4, 4).astype(np.float32)
        assert poses_all.shape[0] == num_frames, f"{poses_all.shape[0]} poses vs {num_frames} rgb frames"

        indices = np.arange(num_frames)
        test_mask = indices % 8 == 0
        frame_indices = indices[test_mask] if split == "test" else indices[~test_mask]

        W, H = int(NATIVE_WIDTH / downsample), int(NATIVE_HEIGHT / downsample)
        self.img_wh = (W, H)
        focal = 0.5 * W  # hfov=90 deg: fx = width / (2*tan(45 deg)) = width/2

        self.intrinsics = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

        pix_min = np.array([0.5, 0.5], dtype=np.float32)
        cam_x = (pix_min[0] - W / 2) / focal
        cam_y = (pix_min[1] - H / 2) / focal
        cam_space_right = torch.tensor([-cam_x, 0.0, 0.0], dtype=torch.float32)
        cam_space_up = torch.tensor([0.0, cam_y, 0.0], dtype=torch.float32)

        self.poses = []
        self.all_cameras = []
        self.all_rgbs = []
        self.all_alphas = []
        self.frame_indices = frame_indices
        for i in frame_indices.tolist():
            c2w = torch.from_numpy(poses_all[i])
            self.poses.append(c2w)
            world_right = torch.einsum("j,kj->k", cam_space_right, c2w[:3, :3])
            world_up = torch.einsum("j,kj->k", cam_space_up, c2w[:3, :3])

            self.all_cameras.append(
                TorchCamera(
                    eye=c2w[:3, 3].pin_memory(),
                    right=world_right.pin_memory(),
                    up=world_up.pin_memory(),
                    width=W,
                    height=H,
                )
            )

            im = Image.open(os.path.join(rgb_dir, f"rgb_{i}.png"))
            if downsample != 1.0:
                im = im.resize((W, H), Image.LANCZOS)
            rgba = np.array(im.convert("RGBA"), dtype=np.float32) / 255.0
            if self.alpha_format_on_disk == "premultiplied":
                alphas = torch.tensor(rgba[..., 3], dtype=torch.float32)
                rgbs = torch.tensor(rgba[..., :3], dtype=torch.float32)
            elif self.alpha_format_on_disk == "straight":
                alphas = torch.tensor(rgba[..., 3], dtype=torch.float32)
                rgbs = torch.tensor(rgba[..., :3] * rgba[..., 3:4], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported alpha format on disk: {self.alpha_format_on_disk}")
            im.close()

            self.all_rgbs.append(rgbs)
            self.all_alphas.append(alphas)

        self.poses = torch.stack(self.poses)
        self.all_rgbs = torch.stack(self.all_rgbs)
        self.all_alphas = torch.stack(self.all_alphas)

        # A real, depth-derived point cloud (build_replica_pointcloud.py),
        # not SfM -- Replica ships no COLMAP reconstruction, but it does
        # ship real depth, so back-projected points serve the identical
        # role init_points_sfm expects: a surface-aligned starting point
        # cloud, which matters a lot in practice (a blind scatter-around-
        # cameras init was tried first and collapsed gsplat's training to
        # 0 Gaussians after its first opacity reset, since too few points
        # ever landed near a real surface to earn lasting opacity).
        pointcloud_path = os.path.join(datadir, "pointcloud.pt")
        if os.path.exists(pointcloud_path):
            cloud = torch.load(pointcloud_path, map_location="cpu", weights_only=True)
            self.points3D = cloud["points"]
            self.points3D_color = cloud["colors"]
        else:
            self.points3D = None
            self.points3D_color = None
        self.all_normals = None

        if use_metric3d:
            raise NotImplementedError("Metric3D not wired up for ReplicaDataset (Replica already ships real depth if needed)")
