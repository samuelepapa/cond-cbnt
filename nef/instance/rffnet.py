from typing import Literal

import numpy as np
import torch
from torch import nn

from nef.instance.neural_field_base import NeuralFieldBase


class RFFNet(NeuralFieldBase):
    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        std: float,
        learnable_coefficients: bool,
        final_act: Literal["sigmoid", "relu"] = "sigmoid",
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
        self.encoding = self.RFF(
            num_in=num_in,
            num_hidden=num_hidden_out,
            learnable_coefficients=learnable_coefficients,
            std=std,
        )

        # Hidden layers
        for i in range(self.num_layers):
            self.linears.append(nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out))
            # self.activations.append(nn.ReLU())
            self.activations.append(nn.GELU())
        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_in, out_features=num_out)

    class RFF(nn.Module):
        def __init__(
            self, num_in: int, num_hidden: int, std: float, learnable_coefficients: bool
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

            # Embedding layer
            if learnable_coefficients:
                self.coefficients = nn.Parameter(
                    std * torch.randn(num_in, num_hidden // 2), requires_grad=True
                )
            else:
                self.coefficients = nn.Parameter(
                    std * torch.randn(num_in, num_hidden // 2), requires_grad=False
                )

        def forward(self, x):
            x_proj = self.pi * x @ self.coefficients
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x):
        x = self.encoding(x)
        for m, a in zip(self.linears, self.activations):
            x = a(m(x))
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
