import torch
from torch import nn

from nef.instance.hashnet import HashNet
from nef.instance.siren import Siren
from nef.instance.mlp import MLP
from nef.instance.rffnet import RFFNet


class LatentSpatialAutoDecoder(nn.Module):
    def __init__(self, num_signals, modulation_dim, spatial_cfg, **kwargs):
        super().__init__()

        # Store model parameters
        self.num_signals = num_signals
        self.modulation_dim = modulation_dim
        self.spatial_cfg = spatial_cfg

        if spatial_cfg.shared_net.do:
            spatial_out_dim = spatial_cfg.shared_net.num_in
        else:
            spatial_out_dim = modulation_dim

        # Create small neural field for every signal
        if spatial_cfg.type == "Hash":
            self.codes = nn.ModuleList(
                HashNet(
                    num_in=3,
                    num_layers=spatial_cfg.num_layers,
                    num_hidden_in=spatial_cfg.num_hidden,
                    num_hidden_out=spatial_cfg.num_hidden,
                    num_out=spatial_out_dim,
                    final_act="none",
                    base_resolution=16,
                    log2_max_params_per_level=19,
                    level_dim=2,
                    num_levels=12,
                )
                for _ in range(num_signals)
            )
        elif spatial_cfg.type == "siren":
            self.codes = nn.ModuleList(
                Siren(
                    num_in=3,
                    num_layers=spatial_cfg.num_layers,
                    num_hidden_in=spatial_cfg.num_hidden,
                    num_hidden_out=spatial_cfg.num_hidden,
                    num_out=spatial_out_dim,
                    omega=0.1,
                    final_act=False,
                )
                for _ in range(num_signals)
            )
        elif spatial_cfg.type == "MLP":
            self.codes = nn.ModuleList(
                MLP(
                    num_in=3,
                    num_layers=spatial_cfg.num_layers,
                    num_hidden_in=spatial_cfg.num_hidden,
                    num_hidden_out=spatial_cfg.num_hidden,
                    num_out=spatial_out_dim,
                    final_act=False,
                )
                for _ in range(num_signals)
            )
        elif spatial_cfg.type == "RFF":
            self.codes = nn.ModuleList(
                RFFNet(
                    num_in=3,
                    num_layers=spatial_cfg.num_layers,
                    num_hidden_in=spatial_cfg.num_hidden,
                    num_hidden_out=spatial_cfg.num_hidden,
                    num_out=spatial_out_dim,
                    std=15.0,
                    learnable_coefficients=False,
                    final_act=False,
                )
                for _ in range(num_signals)
            )

            # Overwrite all encodings to share the same coefficients.
            for rff in self.codes:
                rff.encoding = self.codes[0].encoding

        else:
            raise NotImplementedError(f"Unknown spatial type: {spatial_cfg.type}")
        if spatial_cfg.shared_net.do:
            # Create network to map from code to modulation dim
            self.shared_net = MLP(
                num_in=spatial_cfg.shared_net.num_in,
                num_layers=spatial_cfg.shared_net.num_layers,
                num_hidden_in=spatial_cfg.shared_net.num_hidden,
                num_hidden_out=spatial_cfg.shared_net.num_hidden,
                num_out=modulation_dim,
                final_act=False,
            )
        else:
            self.shared_net = nn.Identity()

        # Normalization layer
        if self.spatial_cfg.norm:
            self.norm = nn.BatchNorm1d(num_features=modulation_dim)
        else:
            self.norm = nn.Identity()

    def get_modulations(self, coords, signal_indices):
        # WARNING: THIS IS A VERY HACKY. DOUBLE CHECK IF PER PATIENT BATCHING IS ENABLED UNDER cfg.training.
        # Get embedding for the patient for which the current coordinates are being evaluated.
        patient_idx = signal_indices[0]

        modulations = self.shared_net(self.codes[patient_idx](coords))
        modulations = self.norm(modulations.view(-1, self.modulation_dim)).view_as(
            modulations
        )
        return modulations

    def apply_modulations(self, activations, modulations):
        return activations + modulations


class LatentSpatialFilmAutoDecoder(LatentSpatialAutoDecoder):
    def __init__(self, num_signals, modulation_dim, spatial_cfg, **kwargs):
        # Store model parameters
        modulation_dim = modulation_dim * 2

        super().__init__(num_signals, modulation_dim, spatial_cfg, **kwargs)

    def get_modulations(self, coords, signal_indices):
        # WARNING: THIS IS A VERY HACKY. DOUBLE CHECK IF PER PATIENT BATCHING IS ENABLED UNDER cfg.training.
        # Get embedding for the patient for which the current coordinates are being evaluated.
        patient_idx = signal_indices[0]

        modulations = self.shared_net(self.codes[patient_idx](coords))
        gamma, beta = torch.chunk(modulations, 2, dim=-1)
        return gamma, beta

    def apply_modulations(self, activations, modulations):
        gamma, beta = modulations
        return activations * gamma + beta


class LatentSpatialConcatAutoDecoder(LatentSpatialAutoDecoder):
    def apply_modulations(self, activations, modulations):
        return torch.cat((activations, modulations), dim=-1)
