import numpy as np
import torch
from torch import nn

from nef.conditional.conditional_neural_fields_base import ConditionalNeuralFieldBase
from nef.instance.garfnet import GARFNet


class ConditionalGARFNet(ConditionalNeuralFieldBase, GARFNet):
    def forward(self, x, signal_indices):
        # Get modulations
        z = self.conditioning.get_modulations(signal_indices)

        # If linear layer, add code to output of linear
        for lin, act in zip(self.linears, self.activations):
            x = lin(x)
            x = self.conditioning.apply_modulations(x, z)
            x = act(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
