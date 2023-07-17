import tomo_projector as ttc
import torch
from torch.autograd import Function


def _add_photon_noise(projections, photon_count, torch_rng):
    noisy_data = torch.poisson(
        torch.exp(-projections) * photon_count, generator=torch_rng
    )
    noisy_data = torch.clamp(noisy_data, min=1.0) / photon_count
    projections = -torch.log(noisy_data)

    return projections


def add_photon_noise(projections, photon_count, torch_rng):
    if photon_count is not None:
        if isinstance(projections, list):
            for i in range(len(projections)):
                projections[i] = _add_photon_noise(
                    torch.tensor(projections[i]), photon_count, torch_rng
                ).numpy()
        elif isinstance(projections, torch.Tensor):
            projections = _add_photon_noise(projections, photon_count, torch_rng)
    else:
        pass
    return projections


class CBProjector(Function):
    @staticmethod
    def forward(
        ctx,
        volume,
        vol_bbox,
        src_points,
        det_centers,
        det_frames,
        det_bbox,
        vol_sz,
        det_sz,
        sampling_step_size,
    ):
        output = ttc.cb_proj_forward(
            volume,
            vol_bbox,
            src_points,
            det_centers,
            det_frames,
            det_bbox,
            det_sz,
            sampling_step_size,
            False,
        )
        ctx.vol_bbox = vol_bbox
        ctx.src_points = src_points
        ctx.det_centers = det_centers
        ctx.det_frames = det_frames
        ctx.det_bbox = det_bbox
        ctx.vol_sz = vol_sz
        ctx.det_sz = det_sz
        ctx.sampling_step_size = sampling_step_size
        return output

    @staticmethod
    def backward(ctx, grad_input):
        vol_bbox = ctx.vol_bbox
        src_points = ctx.src_points
        det_centers = ctx.det_centers
        det_frames = ctx.det_frames
        det_bbox = ctx.det_bbox
        vol_sz = ctx.vol_sz
        sampling_step_size = ctx.sampling_step_size
        output = ttc.cb_backproj_forward(
            grad_input,
            vol_bbox,
            src_points,
            det_centers,
            det_frames,
            det_bbox,
            vol_sz,
            1,
            sampling_step_size,
            False,
        )
        return output, None, None, None, None, None, None, None, None
