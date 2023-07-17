import os
from typing import List, Literal

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load

from nef.instance.neural_field_base import NeuralFieldBase

# Cuda backend
_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(
    name="_hash_encoder",
    extra_cflags=["-O3"],  # '-std=c++17'
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",  # undefine flags, necessary!
    ],  # '-arch=sm_70'
    sources=[
        os.path.join(_src_path, "hash_encoder_src", f)
        for f in [
            "hashencoder.cu",
            "bindings.cpp",
        ]
    ],
)
__all__ = ["_backend"]


class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    # @custom_fwd
    def forward(
        ctx, inputs, embeddings, offsets, base_resolution, calc_grad_inputs=False
    ):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape  # batch size, coord dim
        L = offsets.shape[0] - 1  # level
        C = embeddings.shape[1]  # embedding dim for each level
        H = base_resolution  # base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.zeros(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_forward(
            inputs, embeddings, offsets, outputs, B, D, C, L, H, calc_grad_inputs, dy_dx
        )

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, L * C]

        grad = grad.contiguous()

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_backward(
            grad,
            inputs,
            embeddings,
            offsets,
            grad_embeddings,
            B,
            D,
            C,
            L,
            H,
            calc_grad_inputs,
            dy_dx,
            grad_inputs,
        )

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None
        else:
            return None, grad_embeddings, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_levels: int,
        level_dim: int,
        base_resolution: int,
        log2_hashmap_size: int,
    ):
        """Hash encoder for neural fields.

        Args:
            input_dim (int, optional): Input dimension. Defaults to 3.
            num_levels (int, optional): Number of levels. Defaults to 16.
            level_dim (int, optional): Embedding dimension per level. Defaults to 2.
            base_resolution (int, optional): Base resolution. Defaults to 16.
            log2_hashmap_size (int, optional): Log2 of hashmap size. Defaults to 19.
        """
        super().__init__()

        self.input_dim = input_dim  # coord dims, 2 or 3
        self.num_levels = num_levels  # num levels, each level multiply resolution by 2
        self.level_dim = level_dim  # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size  # log2 of hashmap size, this is log of the max number of parameters per level
        self.base_resolution = base_resolution  # base resolution
        self.output_dim = num_levels * level_dim  # output channels

        if level_dim % 2 != 0:
            print(
                "[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)"
            )

        # allocate parameters
        self.offsets = []
        offset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            resolution = base_resolution * 2**i
            params_in_level = min(
                self.max_params, (resolution + 1) ** input_dim
            )  # limit max number
            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32))

        self.n_params = self.offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.zeros(offset, level_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} H={self.base_resolution} params={self.embeddings.shape}"

    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        if inputs.min().item() < -size or inputs.max().item() > size:
            raise ValueError(
                f"HashGrid encoder: inputs range [{inputs.min().item()}, {inputs.max().item()}] not in [{-size}, {size}]!"
            )

        inputs = (inputs + size) / (2 * size)  # map to [0, 1]

        # print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = hash_encode(
            inputs,
            self.embeddings,
            self.offsets,
            self.base_resolution,
            inputs.requires_grad,
        )
        outputs = outputs.view(prefix_shape + [self.output_dim])

        # print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs


class HashNet(NeuralFieldBase):
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
        skip_conn: List[int] = [],  # empty for no skip connections
    ):
        super().__init__(
            num_in=num_in, num_layers=num_layers, num_hidden_in=num_hidden_in, num_hidden_out=num_hidden_out, num_out=num_out
        )
        self.skip_conn = skip_conn

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
            if i + 1 in skip_conn:
                self.net.append(
                    nn.Linear(
                        in_features=num_hidden_in + num_levels * level_dim,
                        out_features=num_hidden_out,
                    )
                )
            else:
                self.net.append(
                    nn.Linear(in_features=num_hidden_in, out_features=num_hidden_out)
                )
            self.net.append(nn.LeakyReLU())

        # Output layer
        self.final_linear = nn.Linear(in_features=num_hidden_in, out_features=num_out)
        if final_act == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_act == "tanh":
            self.final_activation = nn.Tanh()
        elif final_act == "relu":
            self.final_activation = nn.LeakyReLU()
        elif final_act == "none":
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown final activation {final_act}")

        # print the network 
#        print(self)

    def forward(self, x):
        x = enc_out = self.encoder(x)
        for i, m in enumerate(self.net):
            x = m(x)
            if i in self.skip_conn:
                x = torch.cat([x, enc_out], dim=-1)

        x = self.final_linear(x)
        return x
