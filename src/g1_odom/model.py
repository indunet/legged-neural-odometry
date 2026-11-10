"""Lightweight estimator networks (training side, needs PyTorch).

Input  [B, window, n_features] raw features (normalisation is inside the net).
Output [B, n_out] heading-frame velocities; output names come from metadata.
"""
import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class TCN(nn.Module):
    """Two strided 1-D convolutions, global average pooling, linear head (~8k params)."""

    def __init__(self, n_features, mean, std, channels=24, kernel=5, n_out=3):
        super().__init__()
        self.norm = Normalize(mean, std)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, stride=2), nn.GELU(),
            nn.Conv1d(channels, channels, kernel, stride=2), nn.GELU())
        self.head = nn.Linear(channels, n_out)

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)
        return self.head(self.conv(x).mean(dim=2))


class TCNGaussian(TCN):
    """TCN mean plus a log-variance head on the same pooled features.

    The mean path is the plain TCN, so a TCN checkpoint loads unchanged and
    ``freeze_mean`` keeps its outputs bit-identical while the variance trains.
    """

    def __init__(self, n_features, mean, std, channels=24, kernel=5, n_out=2, var_floor=1e-6):
        super().__init__(n_features, mean, std, channels=channels, kernel=kernel, n_out=n_out)
        self.var_head = nn.Sequential(nn.Linear(channels, 16), nn.GELU(), nn.Linear(16, n_out))
        self.var_floor = float(var_floor)

    def freeze_mean(self):
        for module in (self.norm, self.conv, self.head):
            module.requires_grad_(False)

    def forward(self, x):
        latent = self.conv(self.norm(x).transpose(1, 2)).mean(dim=2)
        var = nn.functional.softplus(self.var_head(latent)) + self.var_floor
        return self.head(latent), torch.log(var)


class GaussianEnsemble(nn.Module):
    """Seeds combined in one graph: mean, mean member variance, spread of member means.

    ``primary`` selects one member's mean instead of the ensemble average; the two
    variances are unchanged.
    """

    def __init__(self, members, primary=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.primary = primary

    def forward(self, x):
        outputs = [member(x) for member in self.members]
        means = torch.stack([o[0] for o in outputs])
        variances = torch.stack([torch.exp(o[1]) for o in outputs])
        average = means.mean(dim=0)
        epistemic = ((means - average) ** 2).mean(dim=0)
        velocity = average if self.primary is None else means[self.primary]
        return velocity, variances.mean(dim=0), epistemic


def build(arch, n_features, mean, std, window, n_out=2, channels=24):
    if arch == "tcn_gauss":
        return TCNGaussian(n_features, mean, std, channels=channels, n_out=n_out)
    if arch == "tcn":
        return TCN(n_features, mean, std, channels=channels, n_out=n_out)
    raise ValueError(arch)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
