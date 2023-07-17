import numpy as np
import torch
from torch import nn

from nef.conditional.conditional_neural_fields_base import ConditionalNeuralFieldBase
from nef.instance.a_rffnet import AdaptiveRFFNet


class ConditionalAdaptiveRFFNet(ConditionalNeuralFieldBase, AdaptiveRFFNet):
    def forward(self, x, signal_indices):
        # Get modulations
        z = self.conditioning.get_modulations(signal_indices)

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
