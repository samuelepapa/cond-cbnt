from functools import partial

import numpy as np
import torch
from torch import nn

from nef.conditional.conditional_neural_fields_base import ConditionalNeuralFieldBase
from nef.instance.siren import Siren


class ConditionalSiren(ConditionalNeuralFieldBase, Siren):
    def forward(self, x, signal_indices):
        # Get modulations
        z = self.conditioning.get_modulations(coords=x, signal_indices=signal_indices)

        for idx, m in enumerate(self.net):
            # If linear layer, add code to output of linear
            if idx > 0 and isinstance(m, nn.Linear):
                x = self.conditioning.apply_modulations(x, z)
                x = m(x)
            else:
                x = m(x)
        x = self.conditioning.apply_modulations(x, z)
        x = self.final_linear(x)
        x = self.sigmoid(x)
        return x
