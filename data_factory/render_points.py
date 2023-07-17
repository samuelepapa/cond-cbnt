import argparse
import logging
import pdb
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tomo_projector_utils.scanner import ConebeamGeometry
from tqdm import tqdm

from rendering.rendering import ConebeamRenderer

log = logging.getLogger(__name__)

if __name__ == "__main__":
    # parse the arguments to get the volumes_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes_dir", type=str, default="./data")
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=25)
    parser.add_argument("--test_size", type=int, default=25)
    args = parser.parse_args()

    volumes_dir = args.volumes_dir

    # find all volumes, just in case
    volume_paths = sorted(
        glob(str(Path(volumes_dir) / "volume_*")),
        key=lambda name: int(Path(name).name.split("_")[1]),
    )

    # select the first volume, all share the same scanner
    volume_path = volume_paths[0]
    g = h5py.File(Path(volume_path) / "volume.h5", "r")

    angles = g["angles"][:]

    geometry_path = Path(volume_path) / "geometry.json"
    geom = ConebeamGeometry.from_json(geometry_path, device="cpu")
    geom.update_angles(angles)

    renderer = ConebeamRenderer(
        src_to_center_dst=geom.source_to_center_dst,
        src_to_det_dst=geom.source_to_detector_dst,
        det_offset=geom.det_offset,
        det_shape=geom.det_dims,
        det_spacing=geom.det_spacing,
    )

    pdb.set_trace()

    with h5py.File(Path(volumes_dir) / "sources_rays.h5", "w") as f:
        f.create_dataset("sources", (len(angles), 3), dtype=np.float32)
        f.create_dataset("rays", (len(angles), 256, 256, 3), dtype=np.float32)

    for i, angle in enumerate(tqdm(angles)):
        source, rays, _ = renderer.compute_rays(angle)
        with h5py.File(Path(volumes_dir) / "sources_rays.h5", "a") as f:
            f["sources"][i] = source
            f["rays"][i] = rays

    log.info("Done.")
