# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Layered hardware-efficient ansatz used as the QCAE latent block (F1.1-F1.5).

Per layer l: ENT_T, then one rotation layer per axis in the rotation set (SNCS Eq. 4 order: CNOT ring, RY, RZ).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import sim

TOPOLOGIES = ("ring", "linear", "all", "none")
ROTSETS = {"YZ": ("RY", "RZ"), "XYZ": ("RX", "RY", "RZ")}


def edges(topology: str, n: int) -> list[tuple[int, int]]:
    """F1.4 CNOT (control, target) list of one entangling layer, in application order."""
    if topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    if topology == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if topology == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if topology == "none":
        return []
    raise ValueError(topology)


@dataclass(frozen=True)
class CircuitSpec:
    n: int
    depth: int
    topology: str
    rotset: str = "YZ"

    def __post_init__(self):
        assert self.n >= 2 and self.depth >= 1
        assert self.topology in TOPOLOGIES and self.rotset in ROTSETS

    @property
    def axes(self) -> tuple[str, ...]:
        return ROTSETS[self.rotset]

    @property
    def n_params(self) -> int:
        """F1.5 P_q = |rotset| * n * p."""
        return len(self.axes) * self.n * self.depth

    def gate_list(self) -> list[tuple]:
        """Variational part only (encoding excluded): ('CNOT', c, t) or ('ROT', axis, layer, qubit)."""
        g = []
        for layer in range(self.depth):
            for c, t in edges(self.topology, self.n):
                g.append(("CNOT", c, t))
            for ax_i, ax in enumerate(self.axes):
                for q in range(self.n):
                    g.append(("ROT", ax, layer, ax_i, q))
        return g

    def n_cnot_raw(self) -> int:
        return sum(1 for x in self.gate_list() if x[0] == "CNOT")

    def n_cnot(self) -> int:
        """F4.2 f2: CNOT count after cancelling adjacent identical CNOT pairs (CNOT.CNOT = I)."""
        return count_cnot_after_cancellation(self.gate_list())

    def n_rot(self) -> int:
        return sum(1 for x in self.gate_list() if x[0] == "ROT") + self.n  # + encoding RY

    def complexity_altares(self) -> float:
        """F4.3 C = (N_rot + N_CNOT)/n  [paper:02 Eq.2], reported only."""
        return (self.n_rot() + self.n_cnot()) / self.n


def count_cnot_after_cancellation(gates: list[tuple]) -> int:
    """Cancel CNOT(c,t) CNOT(c,t) when no gate touching c or t lies between them (repeat until stable)."""
    g = list(gates)
    changed = True
    while changed:
        changed = False
        for i, gi in enumerate(g):
            if gi[0] != "CNOT":
                continue
            qs = {gi[1], gi[2]}
            for j in range(i + 1, len(g)):
                gj = g[j]
                touched = {gj[1], gj[2]} if gj[0] == "CNOT" else {gj[-1]}
                if gj[0] == "CNOT" and gj[1:3] == gi[1:3]:
                    del g[j]
                    del g[i]
                    changed = True
                    break
                if touched & qs:
                    break
            if changed:
                break
    return sum(1 for x in g if x[0] == "CNOT")


def run_circuit(spec: CircuitSpec, y: torch.Tensor, params: torch.Tensor,
                noise: Optional[sim.NoiseModel] = None) -> torch.Tensor:
    """Encode y (B, n) with RY(y_i), apply the ansatz, return <Z_i> (B, n), real float64.

    params shape: (depth, |axes|, n).
    noise None -> exact statevector (F1.3); otherwise density matrix with per-gate channels (F3.6)
    and readout affine map (F3.7).
    """
    b, n = y.shape
    assert n == spec.n
    if noise is None:
        psi = sim.sv_zero(b, n)
        for q in range(n):
            psi = sim.sv_apply_1q(psi, sim.ry(y[:, q]), q)
        for g in spec.gate_list():
            if g[0] == "CNOT":
                psi = sim.sv_apply_cnot(psi, g[1], g[2])
            else:
                _, ax, layer, ax_i, q = g
                psi = sim.sv_apply_1q(psi, sim.GATE_FN[ax](params[layer, ax_i, q]), q)
        return sim.sv_expz(psi)

    rho = sim.dm_zero(b, n).to(sim.CDTYPE)
    S1, S2 = sim.noise_superops(noise)
    for q in range(n):
        rho = sim.dm_apply_1q(rho, sim.ry(y[:, q]), q)
        rho = sim.dm_channel_1q(rho, S1, q)
    for g in spec.gate_list():
        if g[0] == "CNOT":
            c, t = g[1], g[2]
            rho = sim.dm_apply_cnot(rho, c, t)
            rho = sim.dm_channel_2q(rho, S2, c, t)
        else:
            _, ax, layer, ax_i, q = g
            rho = sim.dm_apply_1q(rho, sim.GATE_FN[ax](params[layer, ax_i, q]), q)
            rho = sim.dm_channel_1q(rho, S1, q)
    ez = sim.dm_expz(rho)
    return sim.readout_affine(ez, noise.e0, noise.e1)


def final_state(spec: CircuitSpec, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Statevector (B, 2^n) after encoding + ansatz (used by mechanism measures)."""
    b, n = y.shape
    psi = sim.sv_zero(b, n)
    for q in range(n):
        psi = sim.sv_apply_1q(psi, sim.ry(y[:, q]), q)
    for g in spec.gate_list():
        if g[0] == "CNOT":
            psi = sim.sv_apply_cnot(psi, g[1], g[2])
        else:
            _, ax, layer, ax_i, q = g
            psi = sim.sv_apply_1q(psi, sim.GATE_FN[ax](params[layer, ax_i, q]), q)
    return psi.reshape(b, -1)
