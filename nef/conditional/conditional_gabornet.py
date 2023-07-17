import numpy as np
import torch
from torch import nn

from nef.conditional.conditional_neural_fields_base import ConditionalNeuralFieldBase
from nef.instance.gabornet import GaborNet


class ConditionalGaborNet(ConditionalNeuralFieldBase, GaborNet):
    def forward(self, x, signal_indices):
        # Get modulations
        z = self.conditioning.get_modulations(signal_indices)

        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
            out = self.conditioning.apply_modulations(out, z)
            x = nn.functional.layer_norm(x, x.shape[1:])
        out = self.output_linear(out)
        return out
