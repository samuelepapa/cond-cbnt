from typing import Literal

import numpy as np
import torch
from torch import nn

from nef.instance.neural_field_base import NeuralFieldBase


class MLP(NeuralFieldBase):
    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        final_act: str,
    ):
        super().__init__(
            num_in=num_in, num_layers=num_layers, num_hidden_in=num_hidden_in, num_hidden_out=num_hidden_out, num_out=num_out, final_act=final_act
        )

        # Create network
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Add first hidden layer
        self.first_layer = nn.Linear(in_features=num_in, out_features=num_hidden_in)

        # Hidden layers
        for i in range(self.num_layers):
            self.linears.append(nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out))
            self.activations.append(nn.GELU(approximate='tanh'))

        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_out, out_features=num_out)

    def forward(self, x):
        x = self.first_layer(x)
        for m, a in zip(self.linears, self.activations):
            x = a(m(x))
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
