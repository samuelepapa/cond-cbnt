import numpy as np
import torch
from torch import nn

from nef.conditional.conditional_neural_fields_base import ConditionalNeuralFieldBase
from nef.instance.rffnet import RFFNet


class ConditionalRFFNet(ConditionalNeuralFieldBase, RFFNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize all linear layers such that film conditioning does not impact variance of the output
        for lin in self.linears:
            lin.weight.data *= 1 / np.sqrt(2)
            lin.bias.data *= 1 / np.sqrt(2)
        self.final_linear.weight.data *= 1 / np.sqrt(2)

    def forward(self, x, signal_indices):
        # Get modulations
        z = self.conditioning.get_modulations(coords=x, signal_indices=signal_indices)

        # RFF encoding :-----------------------------------------------)
        x = self.encoding(x)

        # If linear layer, add code to output of linear
        for lin, act in zip(self.linears, self.activations):
            x = lin(x)
            x = self.conditioning.apply_modulations(x, z)
            x = act(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x
