import json
import logging
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt

from tomo_projector_utils.scanner import ConebeamGeometry

from rendering.rendering import ConebeamRenderer

log = logging.getLogger(__name__)


class MultiVolumeConebeamReconDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        volumes_dir,
        original_volumes_dir,
        num_vols=200,
        stage="train",  # TODO(samuele): this is not supported correctly yet
    ):
        self.proj_name = "projections"

        self.volume_dir = volumes_dir
        # select the correct subset of volumes

        self.volume_paths = sorted(
            glob(str(Path(volumes_dir) / "volume_grid_*")),
            key=lambda name: int(Path(name).stem.split("_")[-1]),
        )
        self.original_volume_paths = sorted(
            glob(str(Path(original_volumes_dir) / "volume_*")),
            key=lambda name: int(Path(name).name.split("_")[1]),
        )
        if stage == "train":
            self.volume_paths = self.volume_paths[:num_vols]
            self.original_volume_paths = self.original_volume_paths[:num_vols]
        elif stage == "val":
            self.volume_paths = self.volume_paths[200:225][:num_vols]
            self.original_volume_paths = self.original_volume_paths[200:225][:num_vols]
        elif stage == "test":
            self.volume_paths = self.volume_paths[225:250][:num_vols]
            self.original_volume_paths = self.original_volume_paths[225:250][:num_vols]
        else:
            raise ValueError(
                "stage must be one of train, val, test but is {}".format(stage)
            )

        self.num_patients = len(self.volume_paths)

        self.stage = stage
        self.datasets = []

        # Load the voxel coordinates for each volume
        self.voxel_coords = []
        self.recon_values = []

        # Aggregate number of voxels per volume
        self.voxel_index_to_patient_idx = []
        self.global_voxel_idx_to_patient_voxel_idx = []

        # Store the volume dimensions
        self.volume_shapes = []

        for patient_idx, volume_path in tqdm(enumerate(self.volume_paths)):
            # load the h5py file containing the volume
            dataset = h5py.File(Path(volume_path), "r")
            # load the volume
            self.datasets.append(dataset)

            # For each volume, we need to store the voxel coordinates
            self.voxel_coords.append(np.reshape(dataset["ref_coords"][:], (-1, 3)))
            self.recon_values.append(dataset["volume"][:].flatten())
            self.volume_shapes.append(dataset["volume"].shape)

            # # Store the mapping from voxel index to patient index
            self.voxel_index_to_patient_idx.append(
                np.ones(np.prod(dataset["volume"].shape), dtype=np.int32) * patient_idx
            )

            # # Store the mapping from global voxel index to patient voxel index
            # self.global_voxel_idx_to_patient_voxel_idx.append(
            #     np.arange(np.prod(dataset["ref_coords"].shape), dtype=np.int32)
            # )
        self.datasets = []
        self.angles = []
        self.geometries = []
        self.renderers = []
        for volume_path in self.original_volume_paths:
            f = h5py.File(Path(volume_path) / "volume.h5", "r")
            self.angles.append(f["angles"][:])

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
        # Flatten the lists
        self.voxel_index_to_patient_idx = np.concatenate(
            self.voxel_index_to_patient_idx, axis=0
        )
        self.voxel_coords = np.concatenate(self.voxel_coords, axis=0)
        self.recon_values = np.concatenate(self.recon_values, axis=0)

        self.dim = 3
        self.channels = 1
        self.rendering_bbox = torch.tensor(
            [[-400, -400, -400], [400, 400, 400]], dtype=torch.float32
        )

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

    def __len__(self):
        return len(self.voxel_coords)

    def get_volume(self, patient_idx):
        if self.stage == "train":
            global_patient_idx = patient_idx
        elif self.stage == "val":
            global_patient_idx = patient_idx + 200
        elif self.stage == "test":
            global_patient_idx = patient_idx + 225

        voxel_indices_for_patient = np.argwhere(
            self.voxel_index_to_patient_idx == patient_idx
        ).flatten()
        return (
            self.voxel_coords[voxel_indices_for_patient],
            self.recon_values[voxel_indices_for_patient],
            global_patient_idx,
        )

    def __getitem__(self, idx):
        # Map to "global" patient index
        patient_idx = self.voxel_index_to_patient_idx[idx]
        if self.stage == "train":
            global_patient_idx = patient_idx
        elif self.stage == "val":
            global_patient_idx = patient_idx + 200
        elif self.stage == "test":
            global_patient_idx = patient_idx + 225

        return self.voxel_coords[idx], self.recon_values[idx], global_patient_idx


if __name__ == "__main__":
    ds = MultiVolumeConebeamReconDataset(
        volumes_dir="../data/volumes_recon_pretraining",
        num_vols=200,
        stage="train",
    )

    patient = 0
    pixels, values, patient_idx = ds.get_volume(patient_idx=patient)
    volume = values.reshape(ds.volume_shapes[patient])

    plt.imshow(volume[volume.shape[0] // 2, :, :])
    plt.show()

    print(pixels.shape)
