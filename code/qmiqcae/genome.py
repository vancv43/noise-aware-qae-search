# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Genome decoding and objectives (F4.1-F4.4)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .circuits import CircuitSpec, TOPOLOGIES

DEPTHS = (1, 2, 3, 4)
QUBITS = (4, 6, 8)
ROTS = ("YZ", "XYZ")
WIDTHS = (8, 16, 32)
LR_RANGE = (1e-4, 3e-2)
WD_RANGE = (1e-6, 1e-2)
GENES = ("topology", "depth", "n", "rotset", "lr", "wd", "width")
D = len(GENES)


def _cat(u: float, options):
    """F4.1 categorical decode: equal-width bins of [0,1], u = 1 maps to the last option."""
    i = min(int(u * len(options)), len(options) - 1)
    return options[i]


def _log(u: float, lo: float, hi: float) -> float:
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


@dataclass(frozen=True)
class Phenotype:
    topology: str
    depth: int
    n: int
    rotset: str
    lr: float
    wd: float
    width: int

    @property
    def spec(self) -> CircuitSpec:
        return CircuitSpec(self.n, self.depth, self.topology, self.rotset)

    def key(self) -> tuple:
        """Cache key. lr/wd rounded to 3 significant digits so float noise does not defeat the cache."""
        return (self.topology, self.depth, self.n, self.rotset, float(f"{self.lr:.3g}"), float(f"{self.wd:.3g}"),
                self.width)


def decode(g: np.ndarray) -> Phenotype:
    g = np.clip(np.asarray(g, dtype=float), 0.0, 1.0)
    assert g.shape == (D,)
    return Phenotype(
        topology=_cat(g[0], TOPOLOGIES), depth=_cat(g[1], DEPTHS), n=_cat(g[2], QUBITS), rotset=_cat(g[3], ROTS),
        lr=_log(g[4], *LR_RANGE), wd=_log(g[5], *WD_RANGE), width=_cat(g[6], WIDTHS))


def f2_cnot(ph: Phenotype) -> int:
    """F4.2 two-qubit gate count."""
    return ph.spec.n_cnot()


F2_MAX = CircuitSpec(max(QUBITS), max(DEPTHS), "all").n_cnot()  # 4 * 28 = 112


def scalarize(f1: float, f2: int, lam: float = 0.1) -> float:
    """F4.5 for single-objective baselines: f = f1 + lam * f2 / F2_MAX."""
    return f1 + lam * f2 / F2_MAX
