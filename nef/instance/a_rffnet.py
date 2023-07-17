from typing import Literal

import numpy as np
import torch
from torch import nn

from nef.instance.neural_field_base import NeuralFieldBase


class AdaptiveRFFNet(NeuralFieldBase):
    """ Random Fourier Feature Network with learnable coefficients. """
    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        std: float,
        learnable_coefficients: bool,
        final_act: str,
        init_mask_value: float
    ):
        super().__init__(
            num_in=num_in, num_layers=num_layers, num_hidden_in=num_hidden_in, num_hidden_out=num_hidden_out, num_out=num_out, final_act=final_act
        )

        # Standard deviation of randomly sampled coefficients
        self.std = std

        # Create network
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Add first hidden layer, this maps from encoding to hidden
        self.encoding = self.AdaptiveRFF(
            num_in=num_in,
            num_hidden=num_hidden_out,
            learnable_coefficients=learnable_coefficients,
            std=std,
            init_mask_value=init_mask_value
        )

        # Hidden layers
        for i in range(self.num_layers):
            self.linears.append(nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out))
            # self.activations.append(nn.ReLU())
            self.activations.append(nn.GELU())
        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_in, out_features=num_out)

    class AdaptiveRFF(nn.Module):
        def __init__(
            self, num_in: int, num_hidden: int, std: float, learnable_coefficients: bool, init_mask_value: float,
        ):
            super().__init__()

            # Make sure we have an even number of hidden features.
            assert not num_hidden % 2.0

            # Store parameters
            self.num_in = num_in
            self.num_hidden = num_hidden
            self.learnable_coefficients = learnable_coefficients

            # Store pi
            self.pi = 2 * np.pi

            # Embedding layer, sort coefficients by magnitude
            unsorted_coefficients = std * torch.randn(num_in, num_hidden // 2)
            coefficients = torch.gather(unsorted_coefficients, 1, torch.abs(unsorted_coefficients).sort(dim=1)[1])

            if learnable_coefficients:
                self.coefficients = nn.Parameter(coefficients, requires_grad=True)
            else:
                self.coefficients = nn.Parameter(coefficients, requires_grad=False)

            # Store mask parameter
            self.mask_parameter = nn.Parameter(torch.tensor([init_mask_value]), requires_grad=True)

            # Create linspace from 0 to 1 in number of coefficients steps
            self.register_buffer('mask', torch.linspace(0, 1, self.num_hidden // 2, requires_grad=False).unsqueeze(0))

        def forward(self, x):
            # Compute masked coefficients
            masked_coefficients = torch.exp(-torch.pow(self.mask, 2) / (2 * torch.pow(self.mask_parameter, 2)))

            # Apply mask to coefficients, then compute projection with x.
            x_proj = self.pi * x @ self.coefficients
            return torch.cat([torch.sin(x_proj) * masked_coefficients, torch.cos(x_proj) * masked_coefficients], dim=-1)

    def forward(self, x):
        x = self.encoding(x)
        for m, a in zip(self.linears, self.activations):
            x = a(m(x))
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
