import torch
from torch import nn

from einops import rearrange
import numpy as np


# from https://github.com/boschresearch/multiplicative-filter-networks/blob/main/mfn/mfn.py
class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2*torch.rand(1, 1, out_features, in_features)-1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, 1.0).sample((out_features,)))
        self.linear.weight.data *= weight_scale*self.gamma[:, None]**0.5
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        if len(x.shape) == 2:
            x_r = rearrange(x, 'b d -> b 1 d')
            D = torch.norm(x_r - self.mu.squeeze(0), dim=-1) ** 2
        elif len(x.shape) == 3:
            x_r = rearrange(x, 'b n d -> b n 1 d')
            D = torch.norm(x_r-self.mu, dim=-1)**2
        return torch.sin(self.linear(x)) * torch.exp(-0.5*D*self.gamma[None])


class GaborNet(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        num_layers: int,
        input_scale: float,
        alpha: float
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(num_hidden_in, num_hidden_out) for _ in range(num_layers)]
        )
        self.output_linear = \
            nn.Sequential(nn.Linear(num_hidden_in, num_out),
                          nn.Sigmoid())

        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    num_in,
                    num_hidden_out,
                    input_scale / np.sqrt(num_layers + 1),
                    alpha / (num_layers + 1)
                )
                for _ in range(num_layers + 1)
            ]
        )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i-1](out)
        out = self.output_linear(out)

        return out


# from https://github.com/computational-imaging/bacon/blob/main/modules.py
def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-(12/num_input)**0.5, (12/num_input)**0.5)
