# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Hybrid denoising autoencoder with interchangeable latent blocks (F1.*, F5.*).

All latent blocks map y in [-pi, pi]^n (after BatchNorm + pi*tanh, F1.0) to z in [-1, 1]^n, so the decoder
sees the same range whatever the block.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from . import sim
from .circuits import CircuitSpec, run_circuit


class Encoder(nn.Module):
    """F1.0 SNCS backbone: Conv(1,h,3,s2,p1)-LReLU-Conv(h,2h,3,s2,p1)-LReLU-Linear(2h*7*7, n)."""

    def __init__(self, n: int, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, h, 3, 2, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(h, 2 * h, 3, 2, 1), nn.LeakyReLU(0.1),
            nn.Flatten(), nn.Linear(2 * h * 7 * 7, n))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n: int, h: int):
        super().__init__()
        self.h = h
        self.fc = nn.Linear(n, 2 * h * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * h, h, 3, 2, 1, output_padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(h, 1, 3, 2, 1, output_padding=1), nn.Sigmoid())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 2 * self.h, 7, 7))


# ------------------------------------------------------------------ latent blocks
class QuantumLatent(nn.Module):
    """F1.1-F1.4. frozen=True gives the frozen-random control (F5.3)."""

    def __init__(self, spec: CircuitSpec, frozen: bool = False, init_scale: float = 0.1,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.spec = spec
        shape = (spec.depth, len(spec.axes), spec.n)
        if frozen:
            w = torch.rand(shape, generator=generator, dtype=torch.float64) * 2 * math.pi
            self.register_buffer("params", w)
        else:
            w = torch.randn(shape, generator=generator, dtype=torch.float64) * init_scale  # [paper:04 code]
            self.params = nn.Parameter(w)
        self.noise: Optional[sim.NoiseModel] = None
        self.shots: Optional[int] = None
        self.shot_gen: Optional[torch.Generator] = None

    def forward(self, y):
        rdt = torch.float64 if sim.CDTYPE == torch.complex128 else torch.float32
        z = run_circuit(self.spec, y.to(rdt), self.params.to(rdt), noise=self.noise)
        if self.shots is not None:
            z = sim.sample_shots(z, self.shots, self.shot_gen)
        return z.to(y.dtype)


class MLPLatent(nn.Module):
    """F5.4 z = tanh(W2 tanh(W1 (y/pi) + b1) + b2), hidden k; params 2nk + k + n."""

    def __init__(self, n: int, k: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, k), nn.Tanh(), nn.Linear(k, n), nn.Tanh())

    def forward(self, y):
        return self.net(y / math.pi)


class LinearTanhLatent(nn.Module):
    """F5.5 z = tanh(W (y/pi) + b); params n^2 + n (full feature mixing)."""

    def __init__(self, n: int):
        super().__init__()
        self.lin = nn.Linear(n, n)

    def forward(self, y):
        return torch.tanh(self.lin(y / math.pi))


class IdentityLatent(nn.Module):
    """F5.6 z = y/pi (no latent block)."""

    def forward(self, y):
        return y / math.pi


def mlp_hidden_for(p_target: int, n: int) -> int:
    """F5.4: k = max(n, smallest k with 2nk + k + n >= p_target). k >= n keeps the control full rank
    (k = 1 would squeeze the latent to one dimension and cripple the classical control); the parameter
    count is then >= the circuit's, which is conservative for any quantum claim. Exact counts are reported."""
    k = 1
    while 2 * n * k + k + n < p_target:
        k += 1
    return max(n, k)


class HybridAE(nn.Module):
    def __init__(self, n: int, h: int, latent: nn.Module):
        super().__init__()
        self.encoder = Encoder(n, h)
        self.bn = nn.BatchNorm1d(n)
        self.latent = latent
        self.decoder = Decoder(n, h)

    def forward(self, x):
        y = math.pi * torch.tanh(self.bn(self.encoder(x)))  # F1.0 [paper:04]
        return self.decoder(self.latent(y))


class PlainCAE(nn.Module):
    """Classical CAE reference as in Kit / SNCS (no BN, no tanh)."""

    def __init__(self, n: int, h: int):
        super().__init__()
        self.encoder = Encoder(n, h)
        self.decoder = Decoder(n, h)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DnCNN(nn.Module):
    """Upper classical reference (Zhang et al. 2017 style, residual): depth 17, 64 channels."""

    def __init__(self, depth: int = 17, ch: int = 64):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return (x - self.net(x)).clamp(0, 1)


def build_model(kind: str, n: int, h: int, spec: Optional[CircuitSpec] = None, seed: int = 0) -> nn.Module:
    """kind in {qcae, qcae_frozen, mlp_matched, lintanh, identity, cae, dncnn}."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if kind == "qcae":
        return HybridAE(n, h, QuantumLatent(spec, generator=g))
    if kind == "qcae_frozen":
        return HybridAE(n, h, QuantumLatent(spec, frozen=True, generator=g))
    if kind == "mlp_matched":
        return HybridAE(n, h, MLPLatent(n, mlp_hidden_for(spec.n_params, n)))
    if kind == "lintanh":
        return HybridAE(n, h, LinearTanhLatent(n))
    if kind == "identity":
        return HybridAE(n, h, IdentityLatent())
    if kind == "cae":
        return PlainCAE(n, h)
    if kind == "dncnn":
        return DnCNN()
    raise ValueError(kind)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def set_noise(model: nn.Module, noise: Optional[sim.NoiseModel], shots: Optional[int] = None,
              shot_seed: Optional[int] = None):
    for mod in model.modules():
        if isinstance(mod, QuantumLatent):
            mod.noise = noise
            mod.shots = shots
            mod.shot_gen = None if shot_seed is None else torch.Generator().manual_seed(shot_seed)
