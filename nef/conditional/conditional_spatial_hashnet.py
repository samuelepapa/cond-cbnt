from typing import Literal

from torch import nn

from nef.conditional.conditional_neural_fields_base import ConditionalNeuralFieldBase
from nef.instance.hashnet import HashEncoder
from nef.instance.neural_field_base import NeuralFieldBase


class ConditionalHashNet(ConditionalNeuralFieldBase, NeuralFieldBase):
    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        num_levels: int,
        level_dim: int,
        base_resolution: int,
        log2_max_params_per_level: int,
        final_act: Literal["sigmoid", "tanh", "relu", "none"] = "sigmoid",
        **kwargs
    ):
        super().__init__(
            num_in=num_in,
            num_layers=num_layers,
            num_hidden_in=num_hidden_in,
            num_hidden_out=num_hidden_out,
            num_out=num_out,
            final_act=final_act,
            **kwargs
        )

        # Create network
        self.encoder = HashEncoder(
            input_dim=self.num_in,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_max_params_per_level,
        )

        self.net = nn.ModuleList()

        # Add first hidden layer, this maps from encoding to hidden
        self.net.append(
            nn.Linear(in_features=num_levels * level_dim, out_features=num_hidden_out)
        )
        self.net.append(nn.LeakyReLU())

        # Hidden layers
        for i in range(self.num_layers):
            self.net.append(
                nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out)
            )
            self.net.append(nn.LeakyReLU())

        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_in, out_features=num_out)

        # print the network
        print(self)

    def forward(self, x, signal_indices):
        # Get modulations
        z = self.conditioning.get_modulations(signal_indices)

        if len(x.shape) == 3:
            z = z.unsqueeze(1)

        x = self.encoder(x)
        for i, m in enumerate(self.net):
            x = m(x)
            x = self.conditioning.apply_modulations(x, z)

        x = self.final_linear(x)
        x = self.final_activation(x)

        return x
