import logging
from typing import Literal, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class ConebeamRenderer:
    def __init__(
        self,
        src_to_center_dst: float,
        src_to_det_dst: float,
        det_spacing: np.ndarray,
        det_shape: np.ndarray,
        det_offset: float,
    ):
        self.src_to_center_dst = src_to_center_dst
        self.src_to_det_dst = src_to_det_dst
        self.det_spacing = det_spacing  # [height, width]
        self.det_shape = det_shape  # [height, width]
        self.det_size = self.det_spacing * self.det_shape  # [height, width]
        self.det_offset = det_offset

    def compute_rays(self, phi):
        W, H = self.det_shape[1], self.det_shape[0]
        w_spacing, h_spacing = self.det_spacing[1], self.det_spacing[0]
        x, y = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
        x = x[..., None]
        y = y[..., None]
        det_to_center_dst = self.src_to_det_dst - self.src_to_center_dst

        # [z, y, x]
        z_axis = np.array([1.0, 0.0, 0.0])

        # source of the radiation
        src_pos = np.array(
            [
                0.0,
                np.cos(phi) * self.src_to_center_dst,
                np.sin(phi) * self.src_to_center_dst,
            ]
        )
        # center of the detector
        det_pos = np.array(
            [0.0, -np.cos(phi) * det_to_center_dst, -np.sin(phi) * det_to_center_dst]
        )
        # basis vectors for the detector plane in the 3D space coordinate system
        det_U = z_axis  # along the height
        norm_vec = np.array([0.0, -np.cos(phi), -np.sin(phi)])
        det_V = np.cross(det_U, norm_vec)  # along the width

        # bottom left of the detector in 3D coordinate system
        bottom_left = det_pos - (
            W * 0.5 * det_U * w_spacing
            + H * 0.5 * det_V * h_spacing
            - self.det_offset * det_V
        )

        # the selected point
        point = bottom_left + (
            w_spacing * (x + 0.5) * det_U + h_spacing * (y + 0.5) * det_V
        )

        ray_dir = point - src_pos
        norm_dir = np.linalg.norm(ray_dir, axis=2)[:, :, None]
        ray_dir /= norm_dir

        return src_pos, ray_dir, norm_dir

    def grid_coords(self, vol_bbox, vol_shape):
        """Compute the coordinates of the grid points in the volume bounding box."""
        meshgrid = np.meshgrid(
            np.linspace(vol_bbox[0][0], vol_bbox[1][0], vol_shape[0]),
            np.linspace(vol_bbox[0][1], vol_bbox[1][1], vol_shape[1]),
            np.linspace(vol_bbox[0][2], vol_bbox[1][2], vol_shape[2]),
        )
        return meshgrid

    def sample_points(
        self,
        phi,
        vol_bbox: torch.Tensor,
        norm_bbox: torch.Tensor,
        params: Optional[dict] = None,
        sampling_mode: Literal[
            "constant_step_size", "constant_num_samples"
        ] = "constant_step_size",
        device=torch.device("cpu"),
    ):
        """
        Rays are assumed to be normalized. The step size is the distance between the sampled points.

        Args:
            phi: the angle of the projection, float
            vol_bbox: the bounding box of the volume, shape (2, 3) the first row is the min values
                        (bottom left) and the second row is the max values (top right)
            step_size: the distance between the sampled points, float
            sampling_mode: the sampling mode, can be "constant_step_size"

        Returns:
            sampled_points: the sampled points, shape (H, W, N, 3)
            mask: the mask of the sampled points, shape (H, W, N)
            weights: the weights of the sampled points, shape (H, W, N)

        """
        box_min, box_max = vol_bbox[0], vol_bbox[1]
        sources, rays, _ = self.compute_rays(phi)
        rays = torch.tensor(rays, dtype=torch.float, device=device)
        sources = torch.tensor(sources, dtype=torch.float, device=device)

        tmin, tmax = self.ray_box_intersection(sources, rays, box_min, box_max)
        tmin = tmin[..., None]
        tmax = tmax[..., None]

        if sampling_mode == "constant_step_size":
            line_samples = torch.arange(0, self.src_to_det_dst, params["step_size"]).to(
                device
            )
            step_size = params["step_size"]
        elif sampling_mode == "constant_num_samples":
            line_samples = torch.linspace(0, 1, params["num_samples"]).to(device) * (
                tmax - tmin
            )
            step_size = (tmax - tmin) / params["num_samples"]
        else:
            raise NotImplementedError(f"Sampling mode {sampling_mode} not implemented.")

        samples = tmin + line_samples
        weights = torch.ones_like(samples)
        # step size for each ray can be different
        first_step_size = samples[:, :, 1] - samples[:, :, 0]
        # move inside the box by half a step
        samples = samples + first_step_size[:, :, None] / 2

        # make it into torch tensors
        #        samples = torch.tensor(samples, dtype=torch.float, device=device)

        sampled_points = sources + samples[:, :, :, None] * rays[:, :, None, :]

        if sampling_mode == "constant_step_size":
            # find the points that are inside the volume bounding box
            mask = (sampled_points[:, :, :, 0] >= box_min[0]) & (
                sampled_points[:, :, :, 0] <= box_max[0]
            )
            mask = (
                mask
                & (sampled_points[:, :, :, 1] >= box_min[1])
                & (sampled_points[:, :, :, 1] <= box_max[1])
            )
            mask = (
                mask
                & (sampled_points[:, :, :, 2] >= box_min[2])
                & (sampled_points[:, :, :, 2] <= box_max[2])
            )

            # locate last point in the volume
            diff_mask = torch.zeros_like(mask)
            diff_mask[:, :, :-1] = mask[:, :, 1:]
            last_point_inside = np.logical_xor(diff_mask, mask)
            # set the weight of the last point to 0.5
            weights[last_point_inside] = 0.5
            # set the weight of the first point to 0.5
            weights[:, :, 0] = 0.5
            # set the weights based on the step size
            weights *= step_size
            weights[~mask] = 0.0
        elif sampling_mode == "constant_num_samples":
            mask = torch.ones(
                (
                    sampled_points.shape[0],
                    sampled_points.shape[1],
                    sampled_points.shape[2],
                ),
                dtype=bool,
                device=device,
            )
            weights[:, :, -1] = 0.5
            # set the weight of the first point to 0.5
            weights[:, :, 0] = 0.5
            # set the weights based on the step size
            weights *= step_size

        # normalize points between 0 and 1 inside the volume's bounding box
        extent = norm_bbox[1] - norm_bbox[0]
        sampled_points[:, :, :, 0] = 2 * (sampled_points[:, :, :, 0]) / extent[0]
        sampled_points[:, :, :, 1] = 2 * (sampled_points[:, :, :, 1]) / extent[1]
        sampled_points[:, :, :, 2] = 2 * (sampled_points[:, :, :, 2]) / extent[2]

        return sampled_points.cpu().numpy(), mask, weights

    @staticmethod
    def ray_box_intersection(
        source_point: torch.Tensor,
        ray_vector: torch.Tensor,
        box_min: torch.Tensor,
        box_max: torch.Tensor,
    ):
        ray_vector[torch.where(torch.abs(ray_vector) < 1e-9)] = 1e-9
        iray_vector = 1 / ray_vector
        tmin = (box_min - source_point) * iray_vector
        tmax = (box_max - source_point) * iray_vector
        real_min = torch.minimum(tmin, tmax)
        real_max = torch.maximum(tmin, tmax)
        minimax = torch.minimum(
            torch.minimum(real_max[:, :, 0], real_max[:, :, 1]), real_max[:, :, 2]
        )
        maximin = torch.maximum(
            torch.maximum(real_min[:, :, 0], real_min[:, :, 1]), real_min[:, :, 2]
        )
        return maximin, minimax

    @staticmethod
    def ray_constant_intersection(
        source_point: torch.Tensor,
        ray_vector: torch.Tensor,
        box_min: torch.Tensor,
        box_max: torch.Tensor,
    ):
        ray_vector[torch.where(torch.abs(ray_vector) < 1e-9)] = 1e-9
        iray_vector = 1 / ray_vector
        tmin = (box_min - source_point) * iray_vector
        tmax = (box_max - source_point) * iray_vector

        return torch.min(tmin), torch.max(tmax)

    @staticmethod
    def simple_ray_box_intersection(
        source_point: torch.Tensor,
        ray_vector: torch.Tensor,
        box_min: torch.Tensor,
        box_max: torch.Tensor,
    ):
        ray_vector[torch.where(torch.abs(ray_vector) < 1e-9)] = 1e-9
        iray_vector = 1 / ray_vector
        tmin = (box_min - source_point) * iray_vector
        tmax = (box_max - source_point) * iray_vector
        real_min = torch.minimum(tmin, tmax)
        real_max = torch.maximum(tmin, tmax)
        minimax = torch.minimum(
            torch.minimum(real_max[:, 0], real_max[:, 1]), real_max[:, 2]
        )
        maximin = torch.maximum(
            torch.maximum(real_min[:, 0], real_min[:, 1]), real_min[:, 2]
        )
        return maximin, minimax

    @staticmethod
    def stacked_ray_box_intersection(
        source_point: torch.Tensor,
        ray_vector: torch.Tensor,
        box_min: torch.Tensor,
        box_max: torch.Tensor,
    ):
        ray_vector[torch.where(torch.abs(ray_vector) < 1e-9)] = 1e-9
        iray_vector = 1 / ray_vector
        tmin = (box_min - source_point) * iray_vector
        tmax = (box_max - source_point) * iray_vector
        real_min = torch.minimum(tmin, tmax)
        real_max = torch.maximum(tmin, tmax)
        minimax = torch.minimum(
            torch.minimum(real_max[:, :, 0], real_max[:, :, 1]), real_max[:, :, 2]
        )
        maximin = torch.maximum(
            torch.maximum(real_min[:, :, 0], real_min[:, :, 1]), real_min[:, :, 2]
        )
        return maximin, minimax


if __name__ == "__main__":
    import matplotlib as m
    import matplotlib.pyplot as plt
    from tomo_projector_utils.scanner import ConebeamGeometry

    source_to_detector_dst = 1500
    source_to_center_dst = 1000

    det_spacing = np.array([2, 2])
    det_shape = np.array([256, 256])
    det_offset = 0.0

    vol_shape = np.array([150, 150, 150])
    vol_spacing = np.array([2, 2, 2])
    norm_bbox = np.array([[-256, -256, -256], [256, 256, 256]])
    angles = np.linspace(0, 205, 5) / 180 * np.pi + np.pi / 2

    geom = ConebeamGeometry(
        source_to_center_dst=source_to_center_dst,
        source_to_detector_dst=source_to_detector_dst,
        det_dims=det_shape,
        det_spacing=det_spacing,
        det_offset=det_offset,
        vol_dims=vol_shape,
        vol_spacing=vol_spacing,
        angles=angles,
        device="cpu",
    )

    renderer = ConebeamRenderer(
        src_to_det_dst=geom.source_to_detector_dst,
        src_to_center_dst=geom.source_to_center_dst,
        det_shape=geom.det_dims,
        det_spacing=geom.det_spacing,
        det_offset=geom.det_offset,
    )

    # for phi in np.linspace(0, 205, 400) / 180 * np.pi + np.pi/2

    plt.figure(figsize=(10, 10))
    for phi in angles:
        print(f"phi: {phi}")
        sampled_points, mask, weights = renderer.sample_points(
            phi,
            params={"num_samples": 25},
            vol_bbox=geom.vol_bbox[0].numpy(),
            norm_bbox=norm_bbox,
            sampling_mode="constant_num_samples",
        )
        print(sampled_points[128, 128, :])
        # print(sampled_points.shape)
        # # plot 3d points
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.scatter(
        #     sampled_points[:, 128, :, 0].flatten(),
        #     sampled_points[:, 128, :, 1].flatten(),
        #     sampled_points[:, 128, :, 2].flatten(),
        #     marker=".",
        #     c=weights[:, 128, :].flatten(),
        #     cmap=m.cm.jet,
        #     alpha=1.,
        # )
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.set_xlim(-1500, 1500)
        # ax.set_ylim(-1500, 1500)
        # ax.set_zlim(-1500, 1500)
        # plt.show()

        sel_slice = 64
        sel_vertical = slice(0, 256, 5)
        # color the points based on the weight
        plt.scatter(
            sampled_points[sel_slice, sel_vertical, :, 0][
                mask[sel_slice, sel_vertical]
            ].flatten(),
            sampled_points[sel_slice, sel_vertical, :, 1][
                mask[sel_slice, sel_vertical]
            ].flatten(),
            marker=".",
            c=weights[sel_slice, sel_vertical][mask[sel_slice, sel_vertical]].flatten(),
            cmap=m.cm.jet,
            alpha=0.1,
        )

    plt.xlabel("")
    plt.colorbar()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig("test.png")
