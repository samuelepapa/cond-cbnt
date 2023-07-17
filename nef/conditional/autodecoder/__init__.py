import pdb

from nef.conditional.autodecoder.autodecoder import (
    AutoDecoder,
    LatentAutoDecoder,
    LatentFilmAutoDecoder,
    LatentConcatAutoDecoder,
)
from nef.conditional.autodecoder.spatial_autodecoder import (
    LatentSpatialAutoDecoder,
    LatentSpatialFilmAutoDecoder,
    LatentSpatialConcatAutoDecoder,
)


def get_autodecoder(conditioning_cfg, num_signals, modulation_dim):
    if (
        conditioning_cfg.mode == "additive"
        and conditioning_cfg.latent_conditioning
        and not conditioning_cfg.spatial_conditioning
    ):
        return LatentAutoDecoder(
            num_signals=num_signals,
            code_dim=conditioning_cfg.code_dim,
            modulation_dim=modulation_dim,
        )
    elif (
        conditioning_cfg.mode == "additive"
        and not conditioning_cfg.latent_conditioning
        and not conditioning_cfg.spatial_conditioning
    ):
        return AutoDecoder(num_signals=num_signals, code_dim=modulation_dim)
    elif (
        conditioning_cfg.mode == "film"
        and conditioning_cfg.latent_conditioning
        and not conditioning_cfg.spatial_conditioning
    ):
        return LatentFilmAutoDecoder(
            num_signals=num_signals,
            code_dim=conditioning_cfg.code_dim,
            modulation_dim=modulation_dim,
        )
    elif (
        conditioning_cfg.mode == "additive"
        and conditioning_cfg.latent_conditioning
        and conditioning_cfg.spatial_conditioning
    ):
        return LatentSpatialAutoDecoder(
            num_signals=num_signals,
            code_dim=conditioning_cfg.code_dim,
            modulation_dim=modulation_dim,
            spatial_cfg=conditioning_cfg.spatial_cfg,
        )
    elif (
        conditioning_cfg.mode == "film"
        and conditioning_cfg.latent_conditioning
        and conditioning_cfg.spatial_conditioning
    ):
        return LatentSpatialFilmAutoDecoder(
            num_signals=num_signals,
            code_dim=conditioning_cfg.code_dim,
            modulation_dim=modulation_dim,
            spatial_cfg=conditioning_cfg.spatial_cfg,
        )
    elif (
        conditioning_cfg.mode == "concatenate"
        and conditioning_cfg.latent_conditioning
        and not conditioning_cfg.spatial_conditioning
    ):
        return LatentConcatAutoDecoder(
            num_signals=num_signals,
            code_dim=conditioning_cfg.code_dim,
            modulation_dim=modulation_dim,
        )
    elif (
        conditioning_cfg.mode == "concatenate"
        and conditioning_cfg.latent_conditioning
        and conditioning_cfg.spatial_conditioning
    ):
        return LatentSpatialConcatAutoDecoder(
            num_signals=num_signals,
            code_dim=conditioning_cfg.code_dim,
            modulation_dim=modulation_dim,
            spatial_cfg=conditioning_cfg.spatial_cfg,
        )
    else:
        raise NotImplementedError(
            f"Invalid combination: {conditioning_cfg.mode} + latent: {conditioning_cfg.latent_conditioning}."
        )
