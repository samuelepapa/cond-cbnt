from nef.instance import HashNet, RFFNet, Siren, GARFNet, GaborNet
from nef.conditional import (
    ConditionalHashNet,
    ConditionalRFFNet,
    ConditionalAdaptiveRFFNet,
    ConditionalSiren,
    ConditionalGARFNet,
    ConditionalMLP,
    ConditionalGaborNet,
    ConditionalRFFNet,
    ConditionalAdaptiveRFFNet,
)


def get_neural_field(nef_cfg, num_in, num_out, num_signals=None):
    """Get neural field model as specified by config.

    Args:
        cfg: Config object.
        num_in: Number of input features (dimensionality of volume to reconstruct).
        num_signals: Number of signals (used for conditional nef).
    """
    if not nef_cfg.conditioning.do:
        if nef_cfg.type == "Hash":
            return HashNet(
                num_in=num_in,
                num_hidden_in=nef_cfg.num_hidden,
                num_hidden_out=nef_cfg.num_hidden,
                num_layers=nef_cfg.num_layers,
                num_out=num_out,
                num_levels=nef_cfg.hash.num_levels,
                level_dim=nef_cfg.hash.level_dim,
                base_resolution=nef_cfg.hash.base_resolution,
                log2_max_params_per_level=nef_cfg.hash.log2_max_params_per_level,
                final_act=nef_cfg.final_act,
                skip_conn=nef_cfg.skip_conn,
            )
        elif nef_cfg.type == "RFF":
            return RFFNet(
                num_in=num_in,
                num_hidden_in=nef_cfg.num_hidden,
                num_hidden_out=nef_cfg.num_hidden,
                num_layers=nef_cfg.num_layers,
                num_out=num_out,
                std=nef_cfg.rff.std,
                learnable_coefficients=nef_cfg.rff.learnable_coefficients,
                final_act=nef_cfg.final_act,
            )
        elif nef_cfg.type == "GARF":
            return GARFNet(
                num_in=num_in,
                num_hidden_in=nef_cfg.num_hidden,
                num_hidden_out=nef_cfg.num_hidden,
                num_layers=nef_cfg.num_layers,
                num_out=num_out,
                a=nef_cfg.garf.a,
                trainable=nef_cfg.garf.trainable,
                final_act=nef_cfg.final_act,
            )
        elif nef_cfg.type == "Siren":
            return Siren(
                num_in=num_in,
                num_hidden_in=nef_cfg.num_hidden,
                num_hidden_out=nef_cfg.num_hidden,
                num_layers=nef_cfg.num_layers,
                num_out=num_out,
                omega=nef_cfg.siren.omega,
                final_act=nef_cfg.final_act,
            )
        elif nef_cfg.type == "Gabor":
            return GaborNet(
                num_in=num_in,
                num_hidden_in=nef_cfg.num_hidden,
                num_hidden_out=nef_cfg.num_hidden,
                num_layers=nef_cfg.num_layers,
                num_out=num_out,
                input_scale=nef_cfg.gabor.input_scale,
                alpha=nef_cfg.gabor.alpha,
            )
        else:
            raise NotImplementedError(f"Unrecognized model type: {nef_cfg.type}.")
    else:
        if nef_cfg.conditioning.do:
            # For concatenation conditioning, we need to add the number of input features to the number of signals.
            if nef_cfg.conditioning.mode == "concatenate":
                num_hidden_in = nef_cfg.num_hidden * 2
            else:
                num_hidden_in = nef_cfg.num_hidden
            if nef_cfg.type == "Hash":
                return ConditionalHashNet(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    num_levels=nef_cfg.hash.num_levels,
                    level_dim=nef_cfg.hash.level_dim,
                    base_resolution=nef_cfg.hash.base_resolution,
                    log2_max_params_per_level=nef_cfg.hash.log2_max_params_per_level,
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                )
            elif nef_cfg.type == "Siren":
                return ConditionalSiren(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    omega=nef_cfg.siren.omega,
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                )
            elif nef_cfg.type == "RFF":
                return ConditionalRFFNet(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    std=nef_cfg.rff.std,
                    learnable_coefficients=nef_cfg.rff.learnable_coefficients,
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                )
            elif nef_cfg.type == "aRFF":
                return ConditionalAdaptiveRFFNet(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    std=nef_cfg.rff.std,
                    learnable_coefficients=nef_cfg.rff.learnable_coefficients,
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                    init_mask_value=nef_cfg.arff.init_mask_value,
                )
            elif nef_cfg.type == "GARF":
                return ConditionalGARFNet(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    a=nef_cfg.garf.a,
                    trainable=nef_cfg.garf.trainable,
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                )
            elif nef_cfg.type == "MLP":
                return ConditionalMLP(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                )
            elif nef_cfg.type == "Gabor":
                return ConditionalGaborNet(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    input_scale=nef_cfg.gabor.input_scale,
                    alpha=nef_cfg.gabor.alpha,
                    conditioning_cfg=nef_cfg.conditioning,
                )
            elif nef_cfg.type == "Hash":
                return ConditionalHashNet(
                    num_in=num_in,
                    num_hidden_in=num_hidden_in,
                    num_hidden_out=nef_cfg.num_hidden,
                    num_layers=nef_cfg.num_layers,
                    num_out=num_out,
                    num_signals=num_signals,  # Total number of patients in our dataset.
                    num_levels=nef_cfg.hash.num_levels,
                    level_dim=nef_cfg.hash.level_dim,
                    base_resolution=nef_cfg.hash.base_resolution,
                    log2_max_params_per_level=nef_cfg.hash.log2_max_params_per_level,
                    conditioning_cfg=nef_cfg.conditioning,
                    final_act=nef_cfg.final_act,
                )

    raise NotImplementedError(
        f"Unrecognized combination of conditioning mode and model type: {nef_cfg.conditioning.mode, nef_cfg.type}."
    )
