import torch
from torch import nn

from nef.conditional.autodecoder import get_autodecoder


class ConditionalNeuralFieldBase(nn.Module):
    """Neural Field base class."""

    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        num_signals: int,
        conditioning_cfg: dict,
        **kwargs
    ):
        super().__init__(
            num_in=num_in,
            num_layers=num_layers,
            num_hidden_in=num_hidden_in,
            num_hidden_out=num_hidden_out,
            num_out=num_out,
            **kwargs
        )

        # Store conditioning config
        self.conditioning_cfg = conditioning_cfg

        # Create conditioning module
        self.conditioning = get_autodecoder(
            conditioning_cfg=conditioning_cfg,
            num_signals=num_signals,
            modulation_dim=num_hidden_out,
        )
