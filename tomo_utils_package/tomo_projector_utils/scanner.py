from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch


def circular_conebeam_trajectory(
    src_radius: float,
    det_radius: float,
    src_points: torch.Tensor,
    det_centers: torch.Tensor,
    det_frames: torch.Tensor,
    angles: np.ndarray,
    device: torch.device = torch.device("cuda"),
):
    """
    Computes the positions of the sources, detector center and the vectors that describe the dector plane.
    Rotates the detector along its height (therefore its first dimension).
    The src_points, det_centers and det_frames are modified in place.

    Args:
        src_radius (float): radius of the source
        det_radius (float): radius of the detector
        src_points (torch.Tensor): tensor of shape (batch_size, num_proj, 3) that will be filled with the source positions
        det_centers (torch.Tensor): tensor of shape (batch_size, num_proj, 3) that will be filled with the detector center positions
        det_frames (torch.Tensor): tensor of shape (batch_size, num_proj, 2, 3) that will be filled with the detector frame vectors
        angles (np.ndarray): array of shape (num_proj,) that contains the angles of the projections
        device (torch.device, optional): device on which the tensors are allocated. Defaults to "cuda".

    Returns:
        None
    """
    num_proj = angles.shape[1]
    z_axis = torch.stack(
        (
            torch.ones((num_proj,), device=device, dtype=torch.float32),
            torch.zeros((num_proj,), device=device, dtype=torch.float32),
            torch.zeros((num_proj,), device=device, dtype=torch.float32),
        ),
        dim=1,
    ).to(torch.float32)

    for b in range(angles.shape[0]):
        phi = angles[b].to(device)
        c = torch.cos(phi).to(torch.float32)
        s = torch.sin(phi).to(torch.float32)
        src_points[b] = torch.stack(
            (
                torch.zeros_like(c, device=device, dtype=torch.float32),
                c * src_radius,
                s * src_radius,
            ),
            dim=1,
        )
        det_centers[b] = torch.stack(
            (
                torch.zeros_like(c, device=device, dtype=torch.float32),
                -det_radius * c,
                -det_radius * s,
            ),
            dim=1,
        )
        det_V = torch.cross(
            z_axis,
            torch.stack(
                (torch.zeros_like(c, device=device, dtype=torch.float32), -c, -s), dim=1
            ),
        )
        det_fr = torch.stack([det_V, z_axis], dim=1)
        det_frames[b] = det_fr


@dataclass
class Geometry:
    source_to_detector_dst: float
    source_to_center_dst: float
    vol_dims: np.ndarray  # (z, y, x) aka (depth, height, width)
    det_dims: np.ndarray  # (y, x) aka (height, width)
    det_spacing: np.ndarray  # (y, x)
    vol_spacing: np.ndarray  # (z, y, x)
    det_offset: float = 0
    angles: Optional[Union[np.ndarray, int, torch.Tensor]] = None
    sampling_step_size: float = 0.5

    @classmethod
    def from_json(
        cls, json_file: Union[str, Path], device: Optional[torch.device] = None
    ):
        with open(json_file, "r") as f:
            data = json.load(f)
        if device is not None:
            data["device"] = device

        return cls.from_state_dict(data)

    @classmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def dumps_json(self) -> str:
        return json.dumps(self.get_json())

    def dump_json(self, file: Union[str, Path]):
        with open(file, "w") as f:
            json.dump(self.get_json(), f)

    def get_json(self) -> dict:
        # Create a JSON-safe dictionary
        raise NotImplementedError


@dataclass
class ConebeamGeometry(Geometry):
    device: Union[torch.device, str] = "cuda"

    def __post_init__(self):
        self.vol_real_size = (self.vol_dims * self.vol_spacing).astype(np.float32)
        self.det_real_size = (self.det_dims * self.det_spacing).astype(np.float32)

        # set the angles only at the end, so that everything is updated correctly
        angles = self.angles
        self.angles = None
        self.__compute_bboxes()
        self.__compute_trajectory()
        self.update_angles(angles)

    def state_dict(self):
        return {
            "source_to_center_dst": self.source_to_center_dst,
            "source_to_detector_dst": self.source_to_detector_dst,
            "det_offset": self.det_offset,
            "vol_dims": self.vol_dims,
            "det_dims": self.det_dims,
            "det_spacing": self.det_spacing,
            "vol_spacing": self.vol_spacing,
            "sampling_step_size": self.sampling_step_size,
        }

    def dumps_json(self) -> str:
        return json.dumps(self.get_json())

    def dump_json(self, filepath: Union[Path, str]) -> None:
        with open(filepath, "w") as fp:
            json.dump(self.get_json(), fp)

    def get_json(self) -> dict[str, float | Any]:
        return {
            "source_to_center_dst": self.source_to_center_dst,
            "source_to_detector_dst": self.source_to_detector_dst,
            "det_offset": self.det_offset,
            "vol_dims": self.vol_dims.tolist(),
            "det_dims": self.det_dims.tolist(),
            "det_spacing": self.det_spacing.tolist(),
            "vol_spacing": self.vol_spacing.tolist(),
            "sampling_step_size": self.sampling_step_size,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        required_keys = [
            "source_to_center_dst",
            "source_to_detector_dst",
            "det_offset",
            "vol_dims",
            "det_dims",
            "det_spacing",
            "vol_spacing",
            "sampling_step_size",
        ]
        for key in required_keys:
            if key not in state_dict.keys():
                raise ValueError(
                    f"Key {key} is required in the state_dict for the Geometry."
                )

        convert_keys = [
            "vol_dims",
            "det_dims",
            "det_spacing",
            "vol_spacing",
        ]
        for key in convert_keys:
            state_dict[key] = np.array(state_dict[key])

        return cls(**state_dict)

    def copy(self):
        return self.from_state_dict(self.state_dict())

    def __compute_bboxes(self):
        self._vol_bbox = torch.from_numpy(
            np.stack(
                [
                    -0.5 * self.vol_dims * self.vol_spacing,
                    0.5 * self.vol_dims * self.vol_spacing,
                ],
                axis=0,
            ).astype(np.float32)
        ).to(self.device)
        offset = np.array(
            [
                [self.det_offset, 0],
            ]
        )
        self._det_bbox = torch.from_numpy(
            (
                np.stack(
                    [
                        -0.5 * self.det_dims * self.det_spacing,
                        0.5 * self.det_dims * self.det_spacing,
                    ],
                )
                + offset
            ).astype(np.float32)
        ).to(self.device)

    def __compute_trajectory(self):
        device = self.device
        # update the trajectory only if necessary
        if self.angles is not None:
            if type(self.angles) is int:
                self.angles = torch.tensor(
                    np.linspace(
                        np.pi / self.angles, np.pi + np.pi / self.angles, self.angles
                    )[None, ...],
                    device=self.device,
                    dtype=torch.float32,
                )
            # prepare the tensors in the device for faster computations
            src_points = torch.empty((*self.angles.shape, 3), device=device)
            det_centers = torch.empty((*self.angles.shape, 3), device=device)
            det_frames = torch.empty((*self.angles.shape, 2, 3), device=device)

            # compute the trajectory of the source and the detector
            circular_conebeam_trajectory(
                self.source_to_center_dst,
                self.source_to_detector_dst - self.source_to_center_dst,
                src_points,
                det_centers,
                det_frames,
                angles=self.angles,
                device=device,
            )

            # update the sources, detector centers and frames in the instance of the class
            self.src_points, self.det_centers, self.det_frames = (
                src_points,
                det_centers,
                det_frames,
            )

    def update_angles(
        self,
        angles: Optional[Union[int, np.ndarray, torch.Tensor]],
    ):
        if angles is None:
            return
        if isinstance(angles, np.ndarray):
            angles = torch.tensor(angles, requires_grad=False, dtype=torch.float32)
        elif not isinstance(angles, torch.Tensor):
            raise ValueError("angles must be tensor or ndarray.")

        if self.angles is None or (
            angles is not None
            and (self.angles is not None and not (np.array_equal(angles, self.angles)))
        ):
            if len(angles.shape) == 1:
                # this is necessary when we forget to put the batch dimension in the angle vector
                angles.unsqueeze_(0)
            self.angles = angles
            self.__compute_trajectory()

            # [B, 1, P, 1, 1]
            self.vol_bbox = torch.broadcast_to(
                self._vol_bbox, (self.angles.shape[0], 2, 3)
            )
            self.det_bbox = torch.broadcast_to(
                self._det_bbox, (self.angles.shape[0], 2, 2)
            )

    def update_spacing(self, det_spacing=None, vol_spacing=None):
        assert (
            det_spacing is not None or vol_spacing is not None
        ), "Either det_spacing or vol_spacing must not be None but both are None"

        if det_spacing is not None:
            self.det_spacing = det_spacing
            self.vol_real_size = (self.vol_dims * self.vol_spacing).astype(np.float32)
        if vol_spacing is not None:
            self.vol_spacing = vol_spacing
            self.det_real_size = (self.det_dims * self.det_spacing).astype(np.float32)

        self.__compute_bboxes()

    def update_dims(self, det_dims=None, vol_dims=None):
        assert (
            det_dims is not None or vol_dims is not None
        ), "Either det_dims or vol_dims must not be None, but noth are None"

        if det_dims is not None:
            self.det_dims = det_dims
            self.vol_real_size = (self.vol_dims * self.vol_spacing).astype(np.float32)
        if vol_dims is not None:
            self.vol_dims = vol_dims
            self.det_real_size = (self.det_dims * self.det_spacing).astype(np.float32)

        self.__compute_bboxes()

    def get_projector_params(self, angles=None):
        self.update_angles(angles)
        assert (
            self.angles is not None
        ), "Angles is None, you must provide angles to compute projections"
        return [
            self.vol_bbox,
            self.src_points,
            self.det_centers,
            self.det_frames,
            self.det_bbox,
            self.vol_dims,
            self.det_dims,
            self.sampling_step_size,
        ]

    def get_bp_params(self, angles=None):
        self.update_angles(angles)
        assert (
            self.angles is not None
        ), "Angles is None, you must provide angles to compute back-projections"

        return [
            self.vol_bbox,
            self.src_points,
            self.det_centers,
            self.det_frames,
            self.det_bbox,
            list(self.recon_vol_dims),
            list(self.det_dims),
            0.1,
        ]
