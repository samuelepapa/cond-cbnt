from typing import Literal

from torch import nn

from nef.instance.neural_field_base import NeuralFieldBase
from nef.instance.utils import GaussianActivation, SuperGaussianActivation


class GARFNet(NeuralFieldBase):
    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        a: float,
        trainable: bool,
        final_act: str,
    ):
        super().__init__(
            num_in=num_in, num_layers=num_layers, num_hidden_in=num_hidden_in, num_hidden_out=num_hidden_out, num_out=num_out, final_act=final_act
        )

        # Create network
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()

        # First layer
        self.linears.append(nn.Linear(in_features=num_in, out_features=num_hidden_out))
        self.activations.append(GaussianActivation(a=a, trainable=trainable))

        # Hidden layers
        for i in range(self.num_layers):
            self.linears.append(nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out))
            self.activations.append(GaussianActivation(a=a, trainable=trainable))

        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_in, out_features=num_out)

    def forward(self, x):
        for lin, act in zip(self.linears, self.activations):
            x = lin(x)
            x = act(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
