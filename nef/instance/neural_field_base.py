import torch
from torch import nn


class NeuralFieldBase(nn.Module):
    """Neural Field base class."""

    def __init__(
        self,
        num_in: int,
        num_layers: int,
        num_hidden_in: int,
        num_hidden_out: int,
        num_out: int,
        final_act: str = None,
    ):
        super().__init__()
        # Store model parameters
        self.num_in = num_in
        self.num_layers = num_layers
        self.num_out = num_out
        self.num_hidden_in = num_hidden_in
        self.num_hidden_out = num_hidden_out

        if final_act == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_act == "tanh":
            self.final_activation = nn.Tanh()
        elif final_act == "relu":
            self.final_activation = nn.LeakyReLU()
        elif not final_act:
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown final activation {final_act}")


    def forward(self, coords: torch.Tensor, **kwargs):
        raise NotImplementedError("Neural Field forward pass needs to be implemented.")
