import torch
from torch import nn

import numpy as np


class AutoDecoder(nn.Module):
    def __init__(self, num_signals, code_dim):
        super().__init__()

        # Store model parameters
        self.num_signals = num_signals
        self.code_dim = code_dim

        # Create latent codes / modulations for each signal
        self.codes = nn.Parameter(
            torch.randn(self.num_signals, self.code_dim), requires_grad=True
        )

    def apply_modulations(self, activations, modulations):
        if len(activations.shape) == 3:
            modulations = modulations.unsqueeze(1)
        return activations + modulations

    def get_modulations(self, signal_indices):
        return self.codes[signal_indices]


class LatentAutoDecoder(AutoDecoder):
    def __init__(self, num_signals, code_dim, modulation_dim):
        super().__init__(num_signals=num_signals, code_dim=code_dim)

        # Store model parameters
        self.modulation_dim = modulation_dim

        # Create network that maps from latent modulations to modulations
        self.net = nn.Sequential(
            nn.Linear(in_features=self.code_dim, out_features=self.modulation_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
        )

    def apply_modulations(self, activations, modulations):
        """Apply shift modulations."""
        if len(activations.shape) == 3:
            modulations = modulations.unsqueeze(1)
        return activations + modulations

    def get_modulations(self, signal_indices, **kwargs):
        """Get shift modulations."""
        codes = self.codes[signal_indices]
        return self.net(codes)


class LatentFilmAutoDecoder(AutoDecoder):
    def __init__(self, num_signals, code_dim, modulation_dim):
        super().__init__(num_signals=num_signals, code_dim=code_dim)

        # Store model parameters
        self.modulation_dim = modulation_dim

        # Create network that maps from latent modulations to modulations
        self.shared_net = nn.Sequential(
            nn.Linear(in_features=self.code_dim, out_features=self.modulation_dim),
            nn.GELU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=2 * self.modulation_dim
            ),
            nn.GELU(),
        )

        self.gamma_net = nn.Sequential(
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
            nn.GELU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
            nn.GELU(),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
            nn.GELU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
            nn.GELU(),
        )

        # Initialize last layer of beta_net such that its sum with the activation of the global neural field has
        # a standard deviation of 1.
        self.beta_net[-2].weight.data *= 1 / np.sqrt(2)

    def apply_modulations(self, activations, modulations):
        """Apply FiLM modulations."""
        gammas, betas = modulations
        if len(activations.shape) == 3:
            gammas = gammas.unsqueeze(1)
            betas = betas.unsqueeze(1)
        return activations * gammas + betas

    def get_modulations(self, signal_indices, **kwargs):
        """Get FiLM modulations."""
        codes = self.codes[signal_indices]

        # Split into gamma and beta
        gammas, betas = self.shared_net(codes).chunk(2, dim=-1)

        gammas = self.gamma_net(gammas)
        betas = self.beta_net(betas)
        return gammas, betas


class LatentConcatAutoDecoder(AutoDecoder):
    def __init__(self, num_signals, code_dim, modulation_dim):
        super().__init__(num_signals=num_signals, code_dim=code_dim)

        # Store model parameters
        self.modulation_dim = modulation_dim

        # Create network that maps from latent modulations to modulations
        self.net = nn.Sequential(
            nn.Linear(in_features=self.code_dim, out_features=self.modulation_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.modulation_dim, out_features=self.modulation_dim
            ),
        )

    def apply_modulations(self, activations, modulations):
        """Apply FiLM modulations."""
        return torch.cat((activations, modulations), dim=-1)

    def get_modulations(self, signal_indices, **kwargs):
        """Get shift modulations."""
        codes = self.codes[signal_indices]
        return self.net(codes)
