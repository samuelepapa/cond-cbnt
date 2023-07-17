from functools import partial

import numpy as np
import torch
from torch import nn

from nef.instance.neural_field_base import NeuralFieldBase


class Siren(NeuralFieldBase):
    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        omega: float,
        final_act: str,
    ):
        super().__init__(
            num_in=num_in, num_layers=num_layers, num_hidden_in=num_hidden_in, num_hidden_out=num_hidden_out, num_out=num_out, final_act=final_act
        )

        # Store siren parameters
        self.omega = omega

        # Create network
        self.net = nn.ModuleList()

        # First layer
        self.net.append(nn.Linear(in_features=num_in, out_features=num_hidden_out))
        self.net.append(self.Sine(self.omega))

        # Hidden layers
        for i in range(self.num_layers):
            self.net.append(nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out))
            self.net.append(self.Sine(self.omega))

        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_in, out_features=num_out)
        self.sigmoid = nn.Sigmoid()

        # Apply initialization
        self.net.apply(partial(self.sine_init, omega=self.omega))
        self.final_linear.apply(partial(self.sine_init, omega=self.omega))

        # Apply specialized first layer initialization
        self.net[0].apply(self.first_layer_sine_init)

    class Sine(nn.Module):
        def __init__(self, omega):
            super().__init__()
            self.omega = omega

        def forward(self, x):
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            return torch.sin(self.omega * x)

    @staticmethod
    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, "weight"):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)

    @staticmethod
    def sine_init(m, omega):
        with torch.no_grad():
            if hasattr(m, "weight"):
                num_input = m.weight.size(-1)
                m.weight.uniform_(
                    -np.sqrt(6 / num_input) / omega, np.sqrt(6 / num_input) / omega
                )

    def forward(self, x, **kwargs):
        for m in self.net:
            x = m(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
