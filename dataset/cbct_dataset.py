import json
import logging
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from tomo_projector_utils.scanner import ConebeamGeometry
from tqdm import tqdm

from rendering.rendering import ConebeamRenderer

log = logging.getLogger(__name__)


class SingleVolumeConebeamDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_projs=400, projs_sample_step=10, num_steps=300):
        """

        Args:
            path: path to folder cotaining the volume.npy, projections.npy, angles.json and geometry.json files
            step_size: in millimiters
        """

        geometry_path = Path(path) / "geometry.json"

        f = h5py.File(Path(path) / "volume.h5", "r")

        self.projs = f["projections"][:num_projs:projs_sample_step]
        self.volumes = f["volume"]
        self.angles = f["angles"][:num_projs:projs_sample_step]

        self.geom = ConebeamGeometry.from_json(geometry_path, device="cpu")
        self.geom.update_angles(self.angles)

        self.renderer = ConebeamRenderer(
            src_to_center_dst=self.geom.source_to_center_dst,
            src_to_det_dst=self.geom.source_to_detector_dst,
            det_offset=self.geom.det_offset,
            det_shape=self.geom.det_dims,
            det_spacing=self.geom.det_spacing,
        )
        self.sampled_points, self.mask, self.weights = [], [], []

        self.rendering_bbox = torch.Tensor([[-256, -256, -256], [256, 256, 256]])

        log.info(f"Sampling points...")
        for angle in tqdm(self.angles):
            sampled_points, mask, weights = self.renderer.sample_points(
                angle,
                params={"num_samples": num_steps},
                sampling_mode="constant_num_samples",
                vol_bbox=self.geom.vol_bbox[0],
                norm_bbox=self.rendering_bbox,
            )
            self.sampled_points.append(sampled_points)
            self.mask.append(mask)
            self.weights.append(weights)

        self.dim = 3
        self.channels = 1
        self.num_angles = len(self.angles)

    def __len__(self):
        return self.num_angles * np.prod(self.geom.det_dims)

    def get_whole_volume_coords(self, grid_resolution: int):
        bbox = self.geom.vol_bbox[0].numpy()
        ref_bbox = self.rendering_bbox.numpy()
        extent = bbox / ref_bbox
        x, y, z = np.meshgrid(
            np.linspace(extent[0][0], extent[1][0], grid_resolution, dtype=np.float32),
            np.linspace(extent[0][1], extent[1][1], grid_resolution, dtype=np.float32),
            np.linspace(extent[0][2], extent[1][2], grid_resolution, dtype=np.float32),
        )
        return np.stack([x, y, z], axis=-1)

    def reference_coords(self):
        bbox = self.geom.vol_bbox[0].numpy()
        ref_bbox = self.rendering_bbox.numpy()
        extent = bbox / ref_bbox
        x, y, z = np.meshgrid(
            np.linspace(
                extent[0][0], extent[1][0], self.geom.vol_dims[0], dtype=np.float32
            ),
            np.linspace(
                extent[0][1], extent[1][1], self.geom.vol_dims[1], dtype=np.float32
            ),
            np.linspace(
                extent[0][2], extent[1][2], self.geom.vol_dims[2], dtype=np.float32
            ),
        )
        return np.stack([x, y, z], axis=-1)

    def get_single_angle_params(self, angle_idx):
        return (
            self.sampled_points[angle_idx],
            self.mask[angle_idx],
            self.weights[angle_idx],
        )

    def split_idx(self, idx):
        """
        Let P be the number of projections, H and W be the height and width of the detector.
        Then, idx is in the range [0, P*H*W) and we want to convert it to the projection index, x index and y index.
        There will be H*W indices per projection, so we can use the integer division to get the projection index.
        Then, we can use the modulo operator to get the x and y indices mixed together, we now need to split them.
        The x_y index will be between 0 and H*W, with the row number (y axis) given by the integer division with W,
        while the column number (x axis) given by the modulo operator with W.

        Args:
            idx: the index for the current sample

        Returns:
            proj_idx: the index of the projection
            x_idx: the x index of the detector
            y_idx: the y index of the detector
        """

        proj_idx = idx // np.prod(self.geom.det_dims)
        x_y_idx = idx % np.prod(self.geom.det_dims)
        y_idx = x_y_idx // self.geom.det_dims[1]
        x_idx = x_y_idx % self.geom.det_dims[1]
        return proj_idx, x_idx, y_idx

    def __getitem__(self, idx):
        # select a certain point in the projection.
        proj_idx, x_idx, y_idx = self.split_idx(idx)
        return (
            self.projs[proj_idx, x_idx, y_idx],
            self.sampled_points[proj_idx][
                x_idx, y_idx, :, :
            ],  # (H, W, num. points along ray, 3)
            self.mask[proj_idx][x_idx, y_idx, :],  # (H, W, num. points along ray)
            self.weights[proj_idx][x_idx, y_idx, :],  # (H, W, num. points along ray)
        )


# scaling_coeff = 300
#
# proj_train_mean = 3.8143444437503815
# proj_train_var = 3.5158914645831807
#
# proj_val_mean = 3.800530345916748
# proj_val_var = 3.48624300944366
#
# proj_test_mean = 4.133302541351318
# proj_test_var = 3.8050005435943604
class MultiVolumeConebeamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        volumes_dir,
        num_vols=200,
        num_projs=400,
        projs_sample_step=10,
        num_steps=300,
        stage="train",  # TODO(samuele): this is not supported correctly yet
        noisy_projections=False,
        volume_id=None,
    ):
        self.volume_id = volume_id
        self.noisy_projections = noisy_projections
        if self.noisy_projections:
            self.proj_name = "noisy_projections"
        else:
            self.proj_name = "projections"
        self.num_projs = num_projs
        self.projs_sample_step = projs_sample_step
        self.num_steps = num_steps

        self.geometries = []
        self.renderers = []
        self.angles = []

        self.rendering_bbox = torch.tensor(
            [[-400, -400, -400], [400, 400, 400]], dtype=torch.float32
        )

        self.volume_dir = volumes_dir
        # select the correct subset of volumes
        self.volume_paths = sorted(
            glob(str(Path(volumes_dir) / "volume_*")),
            key=lambda name: int(Path(name).name.split("_")[1]),
        )
        if stage == "train":
            self.volume_paths = self.volume_paths[:num_vols]
        elif stage == "val":
            self.volume_paths = self.volume_paths[200:225][:num_vols]
        elif stage == "test":
            self.volume_paths = self.volume_paths[225:250][:num_vols]

        if volume_id is not None:
            self.volume_paths = [self.volume_paths[volume_id]]

        self.stage = stage

        self.datasets = []
        for volume_path in self.volume_paths:
            f = h5py.File(Path(volume_path) / "volume.h5", "r")
            self.angles.append(f["angles"][: self.num_projs : self.projs_sample_step])

            geometry_path = Path(volume_path) / "geometry.json"
            self.geometries.append(
                ConebeamGeometry.from_json(geometry_path, device=torch.device("cpu"))
            )
            geom = self.geometries[-1]
            geom.update_angles(self.angles[-1])

            self.renderers.append(
                ConebeamRenderer(
                    src_to_center_dst=geom.source_to_center_dst,
                    src_to_det_dst=geom.source_to_detector_dst,
                    det_offset=geom.det_offset,
                    det_shape=geom.det_dims,
                    det_spacing=geom.det_spacing,
                )
            )

        # load the source and the rays
        sources_rays_file = h5py.File(Path(self.volume_dir) / "sources_rays.h5", "r")
        self.sources = sources_rays_file["sources"][
            : self.num_projs : self.projs_sample_step
        ]
        self.rays = sources_rays_file["rays"][: self.num_projs : self.projs_sample_step]
        # parameters for the model to know what to do as input and output
        self.dim = 3
        self.channels = 1

        self.num_patients = len(self.volume_paths)
        self.num_projections = len(self.angles[-1])
        self.pixels_per_projection = np.prod(self.geometries[-1].det_dims)
        self.pixels_per_patient = self.pixels_per_projection * self.num_projections

    def __len__(self):
        return self.num_patients * self.pixels_per_patient

    def reference_coords(self, patient_idx):
        bbox = self.geometries[patient_idx].vol_bbox[0].numpy()
        ref_bbox = self.rendering_bbox.numpy()
        extent = 2 * bbox / (ref_bbox[1] - ref_bbox[0])
        x, y, z = np.meshgrid(
            np.linspace(
                extent[0][0],
                extent[1][0],
                self.geometries[patient_idx].vol_dims[0],
                dtype=np.float32,
            ),
            np.linspace(
                extent[0][1],
                extent[1][1],
                self.geometries[patient_idx].vol_dims[1],
                dtype=np.float32,
            ),
            np.linspace(
                extent[0][2],
                extent[1][2],
                self.geometries[patient_idx].vol_dims[2],
                dtype=np.float32,
            ),
            indexing="ij",
        )
        return np.stack([x, y, z], axis=-1)

    def idx_to_proj_x_y(self, idx):
        patient_idx = idx // self.pixels_per_patient
        proj_idx = idx % self.pixels_per_patient // self.pixels_per_projection
        x_y_idx = idx % self.pixels_per_projection
        y_idx = x_y_idx // self.geometries[-1].det_dims[1]
        x_idx = x_y_idx % self.geometries[-1].det_dims[1]
        return patient_idx, proj_idx, x_idx, y_idx

    def get_volume(self, patient_idx):
        # load the h5py file containing the volume
        dataset = h5py.File(Path(self.volume_paths[patient_idx]) / "volume.h5", "r")
        # load the volume
        volume = dataset["volume"][:]
        return volume

    def __getitem__(self, idx):
        # select a certain point in the projection.
        patient_idx, proj_idx, x_idx, y_idx = self.idx_to_proj_x_y(idx)

        # load the h5py file containing the volume
        dataset = h5py.File(Path(self.volume_paths[patient_idx]) / "volume.h5", "r")
        # load the projections
        projs = dataset[self.proj_name][self.projs_sample_step * proj_idx, x_idx, y_idx]

        # original shape [num. projections, 3]
        source = self.sources[proj_idx, :]
        # original shape [num projections, det height, det width, 3]
        rays = self.rays[proj_idx, x_idx, y_idx, :]

        # Map to "global" patient index
        if self.stage == "train":
            global_patient_idx = patient_idx
        elif self.stage == "val":
            global_patient_idx = patient_idx + 200
        elif self.stage == "test":
            global_patient_idx = patient_idx + 225

        if self.volume_id is not None:
            if self.stage == "train":
                global_patient_idx = self.volume_id
            elif self.stage == "val":
                global_patient_idx = self.volume_id + 200
            elif self.stage == "test":
                global_patient_idx = self.volume_id + 225

        return (
            projs,
            source,
            rays,
            self.geometries[patient_idx].vol_bbox[0],
            global_patient_idx,
            dataset["projections"][self.projs_sample_step * proj_idx, x_idx, y_idx],
        )
