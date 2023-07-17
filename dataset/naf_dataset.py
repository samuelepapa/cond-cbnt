import json
import logging
import pdb
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from tomo_projector_utils.scanner import ConebeamGeometry
from tqdm import tqdm

from rendering.rendering import ConebeamRenderer

log = logging.getLogger(__name__)


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
class SingleVolume(torch.utils.data.Dataset):
    def __init__(
        self,
        volumes_dir,
        name: str = "naf",
        num_projs=400,
        projs_sample_step=10,
        num_steps=300,
        noisy_projections=False,
        norm_const=50,
        num_rays=2048,
        volume_id=0,
    ):
        self.noisy_projections = noisy_projections
        if self.noisy_projections:
            self.proj_name = "noisy_projections"
        else:
            self.proj_name = "projections"
        self.num_projs = num_projs
        self.projs_sample_step = projs_sample_step
        self.num_steps = num_steps

        self.num_rays = num_rays

        if name == "naf":
            self.rendering_bbox = torch.tensor(
                [[-0.164, -0.164, -0.164], [0.164, 0.164, 0.164]], dtype=torch.float32
            )
        elif name == "nt":
            self.rendering_bbox = torch.tensor(
                [[-500, -500, -500], [500, 500, 500]], dtype=torch.float32
            )
        else:
            raise ValueError("Unknown name: {}".format(name))

        self.volume_dir = volumes_dir
        # select the correct subset of volumes
        self.volume_paths = sorted(
            glob(str(Path(volumes_dir) / "volume_*")),
            key=lambda name: int(Path(name).name.split("_")[1]),
        )

        if name == "naf":
            self.volume_path = self.volume_paths[volume_id]
        elif name == "nt":
            self.volume_path = self.volume_paths[volume_id]
        self.volume_id = volume_id
        # load all projections and put them on the GPU
        f = h5py.File(Path(self.volume_path) / "volume.h5", "r")
        self.projs = torch.Tensor(
            f[self.proj_name][: self.num_projs : self.projs_sample_step]
        ).to("cuda")

        self.clean_projs = torch.Tensor(
            f["projections"][: self.num_projs : self.projs_sample_step]
        )

        self.valid_projs = (self.clean_projs > 1e-5).to("cuda")

        self.angles = f["angles"][: self.num_projs : self.projs_sample_step]

        geometry_path = Path(self.volume_path) / "geometry.json"
        self.geometry = ConebeamGeometry.from_json(
            geometry_path, device=torch.device("cpu")
        )
        geom = self.geometry
        geom.update_angles(self.angles)

        # load the source and the rays
        sources_rays_file = h5py.File(Path(self.volume_dir) / "sources_rays.h5", "r")
        self.sources = torch.Tensor(
            sources_rays_file["sources"][: self.num_projs : self.projs_sample_step]
        ).to("cuda")
        self.rays = torch.Tensor(
            sources_rays_file["rays"][: self.num_projs : self.projs_sample_step]
        ).to("cuda")
        # parameters for the model to know what to do as input and output
        self.dim = 3
        self.channels = 1

        self.num_patients = 1
        self.num_projections = len(self.angles)
        self.pixels_per_projection = np.prod(self.geometry.det_dims)
        self.pixels_per_patient = self.pixels_per_projection * self.num_projections

        self.norm_const = 1.0 / norm_const

        W, H = self.geometry.det_dims[1], self.geometry.det_dims[0]
        coords = np.stack(
            np.meshgrid(
                np.linspace(0, W - 1, W), np.linspace(0, H - 1, H), indexing="ij"
            ),
            -1,
        )
        self.coords = torch.Tensor(np.reshape(coords, [-1, 2])).to("cuda")

    def __len__(self):
        return self.num_patients * self.num_projections

    def reference_coords(
        self,
    ):
        bbox = self.geometry.vol_bbox[0].numpy().copy()
        bbox[0] = bbox[0] + self.geometry.vol_spacing[0] / 2
        bbox[1] = bbox[1] - self.geometry.vol_spacing[0] / 2
        ref_bbox = self.rendering_bbox.numpy()
        extent = 2 * bbox / (ref_bbox[1] - ref_bbox[0])
        x, y, z = np.meshgrid(
            np.linspace(
                extent[0][0],
                extent[1][0],
                self.geometry.vol_dims[0],
                dtype=np.float32,
            ),
            np.linspace(
                extent[0][1],
                extent[1][1],
                self.geometry.vol_dims[1],
                dtype=np.float32,
            ),
            np.linspace(
                extent[0][2],
                extent[1][2],
                self.geometry.vol_dims[2],
                dtype=np.float32,
            ),
            indexing="ij",
        )
        # pdb.set_trace()
        return np.stack([x, y, z], axis=-1)

    def idx_to_proj(self, idx):
        patient_idx = idx // self.num_projections
        proj_idx = idx % self.num_projections
        return patient_idx, proj_idx

    def get_volume(
        self,
    ):
        # load the h5py file containing the volume
        dataset = h5py.File(Path(self.volume_path) / "volume.h5", "r")
        # load the volume
        volume = dataset["volume"][:]
        return volume

    def __getitem__(self, idx):
        # select a certain point in the projection.
        patient_idx, proj_idx = self.idx_to_proj(idx)

        # load the projections
        projs = self.projs[proj_idx]

        # select random coordinates
        coords_valid = self.coords[self.valid_projs[proj_idx].flatten()]
        ray_idxs = np.random.choice(
            coords_valid.shape[0], size=(self.num_rays), replace=False
        )
        selected_idxs = coords_valid[ray_idxs].to(torch.int64)

        # select the ray
        projs = projs[selected_idxs[:, 0], selected_idxs[:, 1]].flatten()
        source = self.sources[proj_idx, :][None, ...]
        # original shape [num projections, det height, det width, 3]
        rays = self.rays[proj_idx]
        # select the rays
        rays = torch.reshape(rays[selected_idxs[:, 0], selected_idxs[:, 1], :], (-1, 3))
        return (
            projs,
            source,
            rays,
            self.geometry.vol_bbox[0][None, ...],
            patient_idx,
        )
