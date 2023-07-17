import json
import logging
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from tomo_projector_utils.scanner import ConebeamGeometry
from tqdm import tqdm

import os
import argparse

from rendering.rendering import ConebeamRenderer

log = logging.getLogger(__name__)


def reference_coords(patient_idx):
    bbox = geometries[patient_idx].vol_bbox[0].numpy()
    ref_bbox = rendering_bbox.numpy()
    extent = 2 * bbox / (ref_bbox[1] - ref_bbox[0])
    x, y, z = np.meshgrid(
        np.linspace(
            extent[0][0],
            extent[1][0],
            geometries[patient_idx].vol_dims[0],
            dtype=np.float32,
        ),
        np.linspace(
            extent[0][1],
            extent[1][1],
            geometries[patient_idx].vol_dims[1],
            dtype=np.float32,
        ),
        np.linspace(
            extent[0][2],
            extent[1][2],
            geometries[patient_idx].vol_dims[2],
            dtype=np.float32,
        ),
        indexing="ij",
    )
    return np.stack([x, y, z], axis=-1)


def get_volume(patient_idx):
    # load the h5py file containing the volume
    dataset = h5py.File(Path(volume_paths[patient_idx]) / "volume.h5", "r")
    # load the volume
    volume = dataset["volume"][:]
    return volume


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes_dir", type=str, default="../data/volumes")
    parser.add_argument(
        "--recons_dir", type=str, default="../data/volumes_recon_pretraining"
    )
    args = parser.parse_args()

    proj_name = "projections"

    geometries = []
    renderers = []
    angles = []

    rendering_bbox = torch.tensor(
        [[-400, -400, -400], [400, 400, 400]], dtype=torch.float32
    )

    volume_dir = args.volumes_dir
    # select the correct subset of volumes
    volume_paths = sorted(
        glob(str(Path(args.volumes_dir) / "volume_*")),
        key=lambda name: int(Path(name).name.split("_")[1]),
    )
    datasets = []

    # Create directory for the dataset
    os.makedirs(args.recons_dir, exist_ok=True)

    # Create new h5py file for the dataset
    for patient_idx, volume_path in enumerate(tqdm(volume_paths)):
        f = h5py.File(Path(volume_path) / "volume.h5", "r")
        angles.append(f["angles"][:400:8])

        geometry_path = Path(volume_path) / "geometry.json"
        geometries.append(
            ConebeamGeometry.from_json(geometry_path, device=torch.device("cpu"))
        )
        geom = geometries[-1]
        geom.update_angles(angles[-1])

        renderers.append(
            ConebeamRenderer(
                src_to_center_dst=geom.source_to_center_dst,
                src_to_det_dst=geom.source_to_detector_dst,
                det_offset=geom.det_offset,
                det_shape=geom.det_dims,
                det_spacing=geom.det_spacing,
            )
        )

        ref_coords = reference_coords(patient_idx)
        volume = get_volume(patient_idx)

        # Create h5py file for the volume
        new_file = h5py.File(
            Path(args.recons_dir) / f"volume_grid_{patient_idx}.h5", "w"
        )

        # Create dataset for the volume
        volume_dataset = new_file.create_dataset(
            "volume", shape=volume.shape, dtype=np.float32
        )
        volume_dataset[:] = volume

        # Create dataset for the reference coordinates
        ref_coords_dataset = new_file.create_dataset(
            "ref_coords", shape=ref_coords.shape, dtype=np.float32
        )
        ref_coords_dataset[:] = ref_coords
