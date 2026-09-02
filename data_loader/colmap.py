import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pycolmap

from powerfoam.camera import TorchCamera
from .metric3D import Metric3DEstimator


def image_plane_basis(camera):
    pix_min = np.array([0.5, 0.5], dtype=np.float32)
    ip_min = camera.cam_from_img(pix_min)
    right = np.array([-ip_min[0], 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, ip_min[1], 0.0], dtype=np.float32)
    return torch.from_numpy(right), torch.from_numpy(up)


def get_cam_raymaps(camera):
    x = np.arange(camera.width, dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate([ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1)
    ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


class COLMAPDataset:
    def _maybe_resize(self, im):
        """Downscale to `max_image_width`, reproducing INRIA 3DGS's `--resolution -1` rule.

        Theirs is: `global_down = orig_w / 1600` when `orig_w > 1600` (else 1), then
        `resize((int(orig_w / global_down), int(orig_h / global_down)))` with PIL's default
        resampling. Reproduced exactly, including the truncation, so a run can be compared
        against a published 3DGS baseline at the resolution that baseline actually trained at.
        Images at or below the cap are returned untouched.
        """
        if not self.max_image_width or im.width <= self.max_image_width:
            return im
        scale = im.width / self.max_image_width
        return im.resize((int(im.width / scale), int(im.height / scale)))

    def __init__(
        self,
        datadir,
        split,
        downsample,
        alpha_format_on_disk,
        use_metric3d,
        max_image_width=None,
    ):
        assert downsample in [1, 2, 4, 8]
        self.max_image_width = max_image_width

        self.root_dir = datadir
        self.colmap_dir = os.path.join(datadir, "sparse/0/")
        self.split = split
        self.downsample = downsample
        self.alpha_format_on_disk = alpha_format_on_disk
        self.use_metric3d = use_metric3d

        if downsample == 1:
            images_dir = os.path.join(datadir, "images")
        else:
            images_dir = os.path.join(datadir, f"images_{downsample}")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory {images_dir} not found")

        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)

        if len(self.reconstruction.cameras) > 1:
            raise ValueError("Multiple cameras are not supported")

        names = sorted(im.name for im in self.reconstruction.images.values())
        indices = np.arange(len(names))

        metric3d_dir = os.path.join(datadir, "metric3d")
        if use_metric3d and not os.path.exists(metric3d_dir):
            print("Precomputed Metric3D data not found; running now...")
            os.makedirs(metric3d_dir)
            input_dir = os.path.join(datadir, "images")
            input_paths = list(os.path.join(input_dir, name) for name in names)
            estimator = Metric3DEstimator()
            estimator.process_dir(input_paths, metric3d_dir)

        if split == "all":
            names = list(names)
        elif split == "train":
            names = list(np.array(names)[indices % 8 != 0])
        elif split == "test":
            names = list(np.array(names)[indices % 8 == 0])
        else:
            raise ValueError(f"Invalid split: {split}")

        names = list(str(name) for name in names)

        im = Image.open(os.path.join(images_dir, names[0]))
        self.img_wh = self._maybe_resize(im).size
        im.close()

        self.camera = list(self.reconstruction.cameras.values())[0]
        self.camera.rescale(self.img_wh[0], self.img_wh[1])

        cam_space_right, cam_space_up = image_plane_basis(self.camera)
        # Depends only on the intrinsics, so it is built once and SHARED by every camera below
        # rather than being expanded into a per-view world-space map. See the note in the camera
        # construction loop.
        cam_ray_dirs = get_cam_raymaps(self.camera).pin_memory()
        self.cam_ray_dirs = cam_ray_dirs

        self.images = []
        for name in names:
            image = None
            for image_id in self.reconstruction.images:
                image = self.reconstruction.images[image_id]
                if image.name == name:
                    break

            if image is None:
                raise ValueError(f"Image {name} not found in COLMAP reconstruction")

            self.images.append(image)

        self.poses = []
        self.all_cameras = []
        # Preallocated PINNED, and filled in place below, instead of appending to a list and then
        # doing `torch.stack(...).pin_memory()`. That sequence holds three copies of the whole
        # image set at once (the list, the stacked tensor, and the pinned tensor), so peak host
        # memory was ~3x the steady-state cost -- the difference between fitting and being
        # OOM-killed on the 733-view scene at full resolution. `DataHandler` calls `.pin_memory()`
        # on these again, which is a no-op returning self for already-pinned tensors.
        n_views = len(self.images)
        W_img, H_img = self.img_wh
        self.all_rgbs = torch.empty((n_views, H_img, W_img, 3), dtype=torch.float32).pin_memory()
        self.all_alphas = torch.empty((n_views, H_img, W_img), dtype=torch.float32).pin_memory()
        for view_idx, image in enumerate(tqdm(self.images)):
            c2w = torch.tensor(
                image.cam_from_world().inverse().matrix(), dtype=torch.float32
            )
            self.poses.append(c2w)
            world_right = torch.einsum("j,kj->k", cam_space_right, c2w[:, :3])
            world_up = torch.einsum("j,kj->k", cam_space_up, c2w[:, :3])

            # The world-space (H, W, 6) ray map is NOT materialised here. It used to be, and
            # pinned, which made host memory the binding constraint on large scenes: the map is
            # 6 floats per pixel, i.e. twice the RGB image, and pinned pages cannot be swapped or
            # reclaimed. A 733-view ScanNet++ scene stalled at a flat 41.9 GB against a 50 GB
            # cgroup -- alive but making no progress, thrashing in reclaim -- and at full
            # resolution the same scene was OOM-killed outright, which ends the SLURM job and
            # returns the node.
            #
            # `TorchCamera` rebuilds the map on demand from `cam_ray_dirs` (shared) and this
            # view's rotation, with arithmetic identical to what was here before, so nothing about
            # the geometry changes. Storage for this term drops from O(views * H * W) to O(H * W).
            self.all_cameras.append(
                TorchCamera(
                    eye=c2w[:3, 3].pin_memory(),
                    right=world_right.pin_memory(),
                    up=world_up.pin_memory(),
                    width=self.img_wh[0],
                    height=self.img_wh[1],
                    cam_ray_dirs=cam_ray_dirs,
                    c2w_rot=c2w[:, :3].contiguous().pin_memory(),
                )
            )

            im = self._maybe_resize(Image.open(os.path.join(images_dir, image.name)))
            # Decoded as uint8 and widened straight into the preallocated destination, rather than
            # building `np.array(..., dtype=np.float32) / 255.0` first. That produced two
            # full-resolution float32 RGBA temporaries per image (~33 MB each at 1752x1168), and
            # the resulting allocator churn -- not the retained data -- was what pushed host memory
            # to the cgroup limit: RSS climbed ~63 MB per image loaded while the retained buffers
            # accounted for only 24 GB. Staying in uint8 until the copy makes the transient 8 MB.
            # `Tensor.copy_` casts uint8 -> float32 during the copy, so no wide temporary exists.
            rgba_u8 = torch.from_numpy(np.asarray(im.convert("RGBA"), dtype=np.uint8))
            dst_rgb, dst_a = self.all_rgbs[view_idx], self.all_alphas[view_idx]
            if self.alpha_format_on_disk == "premultiplied":
                dst_a.copy_(rgba_u8[..., 3]).div_(255.0)
                dst_rgb.copy_(rgba_u8[..., :3]).div_(255.0)
            elif self.alpha_format_on_disk == "straight":
                dst_a.copy_(rgba_u8[..., 3]).div_(255.0)
                dst_rgb.copy_(rgba_u8[..., :3]).div_(255.0).mul_(dst_a.unsqueeze(-1))
            else:
                raise ValueError(
                    f"Unsupported alpha format on disk: {self.alpha_format_on_disk}"
                )
            im.close()
            del rgba_u8, dst_rgb, dst_a

        self.poses = torch.stack(self.poses)

        self.points3D = []
        self.points3D_color = []
        for point in self.reconstruction.points3D.values():
            self.points3D.append(point.xyz)
            self.points3D_color.append(point.color)

        self.points3D = torch.tensor(np.array(self.points3D), dtype=torch.float32)
        self.points3D_color = torch.tensor(
            np.array(self.points3D_color), dtype=torch.float32
        )
        self.points3D_color = self.points3D_color / 255.0

        if use_metric3d:
            self.all_normals = []

            for i, image in enumerate(tqdm(self.images)):
                m3d_name = os.path.splitext(image.name)[0] + ".pt"
                m3d = torch.load(os.path.join(metric3d_dir, m3d_name))
                depth = F.interpolate(
                    m3d["depth"][None, None],
                    size=(self.img_wh[1], self.img_wh[0]),
                    mode="nearest",
                )[0, 0]
                normal = F.interpolate(
                    m3d["normal"][None],
                    size=(self.img_wh[1], self.img_wh[0]),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                normal = normal.permute(1, 2, 0)
                confidence = F.interpolate(
                    m3d["confidence"][None, None],
                    size=(self.img_wh[1], self.img_wh[0]),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                self.all_normals.append(normal)

            self.all_normals = torch.stack(self.all_normals)

        else:
            self.all_normals = None
