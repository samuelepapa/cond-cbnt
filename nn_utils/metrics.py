import logging
from math import exp
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.autograd import Variable

log = logging.getLogger(__name__)


def rmse(volume1: torch.Tensor, volume2: torch.Tensor, fov=None) -> torch.Tensor:
    """
    Computes the root mean square error between two torch Tensors.
    When the two tensors are both in GPU, the value returned is a Tensor with a
    single value inside and stored in the GPU. When they are both in the CPU
    then a float is returned. Pytorch determines this behavior.
    Args:
        - volume1 (torch.Tensor): the first tensor used in the calculation.
        - volume2 (torch.Tensor): the second tensor used in the calculation.
            The operation is commutative.
    Returns:
        (torch.Tensor) with the root mean square error between volume1 and
         volume2: sqrt(mean((volume1-volume2)**2)).
    """
    if fov is not None:
        return torch.sqrt(
            torch.sum(torch.pow(volume1 - volume2, 2) * fov) / torch.sum(fov)
        )
    else:
        return torch.sqrt(torch.mean(torch.pow(volume1 - volume2, 2)))


def mae(volume1: torch.Tensor, volume2: torch.Tensor, fov=None) -> torch.Tensor:
    """
    Computes the mean absolute error between two torch Tensors.
    When the two tensors are both in GPU, the value returned is a Tensor with a
    single value inside and stored in the GPU. When they are both in the CPU
    then a float is returned. Pytorch determines this behavior.
    Args:
        - volume1 (torch.Tensor): the first tensor used in the calculation.
        - volume2 (torch.Tensor): the second tensor used in the calculation.
            The operation is commutative.
    Returns:
        (torch.Tensor) with the mean absolute error between volume1 and
         volume2: mean(abs(volume1-volume2)).
    """
    if fov is not None:
        return torch.sum(torch.abs(volume1 - volume2) * fov) / torch.sum(fov)
    else:
        return torch.mean(torch.abs(volume1 - volume2))


def psnr(volume: torch.Tensor, ground_truth: torch.Tensor, fov=None) -> torch.Tensor:
    """
    Computes the Peak Signal to Noise Ration (PSNR). Peak signal found from ground_truth
    and noise is given as the mean square error (MSE) between volume and ground_truth.
    The value is returned in decibels.
    Args:
        - volume (torch.Tensor): the first tensor used in the calculation.
        - ground_truth (torch.Tensor): the second tensor used in the calculation.
    Returns:
        (torch.Tensor) with the mean of the PNSR of the volume according to the peak
        signal that can be obtained in ground_truth: mean(log10(peak_signal**2/MSE(volume-ground_truth))).
    """
    assert np.equal(volume.shape, ground_truth.shape).all()
    batch_size = volume.shape[0]

    # change the view to compute the metric in a batched way correctly
    w_volume = volume.view(batch_size, -1)
    w_ground_truth = ground_truth.view(batch_size, -1)

    # measure of the noise
    if fov is not None:
        fov = fov.view(batch_size, -1)

        # peak signal from the ground truth
        maxval = torch.max(w_ground_truth * fov, dim=1)[0]

        mse = torch.sum(((w_volume - w_ground_truth) ** 2) * fov, dim=1) / torch.sum(
            fov, dim=1
        )
        return torch.mean(20.0 * torch.log10(maxval) - 10.0 * torch.log10(mse))
    else:
        # peak signal from the ground truth
        maxval = torch.max(w_ground_truth, dim=1)[0]

        mse = torch.mean(((w_volume - w_ground_truth) ** 2), dim=1)

        return torch.mean(20.0 * torch.log10(maxval) - 10.0 * torch.log10(mse))


class SSIMLoss(torch.nn.Module):
    """
    SSIM loss module.
    """

    def __init__(
        self,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        dim: int = 3,
        device: str = "cuda",
    ):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            dim: dimension of the tensors used in the calculation can be 2 or 3.
            device: device where the calculation will be performed.
        """
        super().__init__()
        self.win_size = win_size
        self.dim = dim
        self.device = device
        self.k1, self.k2 = k1, k2
        if dim == 2:
            self.register_buffer(
                "w",
                torch.ones(1, 1, win_size, win_size, device=torch.device(device))
                / win_size**2,
            )
        elif dim == 3:
            self.register_buffer(
                "w",
                torch.ones(
                    1, 1, win_size, win_size, win_size, device=torch.device(device)
                )
                / win_size**3,
            )
        else:
            raise ValueError("Wrong value for dimension")
        NP = win_size**dim
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        fov: Optional[torch.Tensor] = None,
    ):
        assert isinstance(self.w, torch.Tensor)

        #        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        if self.dim == 2:
            conv_op = F.conv2d
        else:
            conv_op = F.conv3d
        ux = conv_op(
            X, self.w, padding=0 if fov is None else self.win_size // 2
        )  # typing: ignore
        uy = conv_op(Y, self.w, padding=0 if fov is None else self.win_size // 2)  #
        uxx = conv_op(X * X, self.w, padding=0 if fov is None else self.win_size // 2)
        uyy = conv_op(Y * Y, self.w, padding=0 if fov is None else self.win_size // 2)
        uxy = conv_op(X * Y, self.w, padding=0 if fov is None else self.win_size // 2)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if fov is None:
            return S.mean()
        else:
            return (S * fov).sum() / torch.sum(fov)
