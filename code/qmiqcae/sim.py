# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Batched statevector and density-matrix simulation in PyTorch (autograd-differentiable).

Qubit ordering: qubit 0 is the most significant bit of the basis index (|q0 q1 ... q_{n-1}>).
The Qiskit cross-checks in tests/ reverse the qubit order to match Qiskit's little-endian convention.

Formula IDs refer to the equations of the study design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

CDTYPE = torch.complex128


# ---------------------------------------------------------------- single-qubit gate matrices (F1.2)
def rx(theta: torch.Tensor) -> torch.Tensor:
    """R_X(theta) = exp(-i theta X / 2); theta shape (...,) -> (..., 2, 2)."""
    t = theta.to(torch.float64) / 2
    c = torch.cos(t).to(CDTYPE)
    s = torch.sin(t).to(CDTYPE)
    m = torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)
    return m


def ry(theta: torch.Tensor) -> torch.Tensor:
    t = theta.to(torch.float64) / 2
    c = torch.cos(t).to(CDTYPE)
    s = torch.sin(t).to(CDTYPE)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rz(theta: torch.Tensor) -> torch.Tensor:
    t = theta.to(torch.float64) / 2
    em = torch.exp(-1j * t.to(CDTYPE))
    ep = torch.exp(1j * t.to(CDTYPE))
    z = torch.zeros_like(em)
    return torch.stack([torch.stack([em, z], -1), torch.stack([z, ep], -1)], -2)


GATE_FN = {"RX": rx, "RY": ry, "RZ": rz}

I2 = torch.eye(2, dtype=CDTYPE)
PX = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
PY = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE)
PZ = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE)


# ---------------------------------------------------------------- statevector (F1.3)
def sv_zero(batch: int, n: int) -> torch.Tensor:
    psi = torch.zeros((batch,) + (2,) * n, dtype=CDTYPE)
    psi.view(batch, -1)[:, 0] = 1.0
    return psi


def sv_apply_1q(psi: torch.Tensor, u: torch.Tensor, q: int) -> torch.Tensor:
    """psi (B, 2,...,2); u (2,2) or (B,2,2); apply on qubit q."""
    ax = q + 1
    x = torch.movedim(psi, ax, -1)  # (B, ..., 2)
    if u.dim() == 2:
        x = x @ u.transpose(-1, -2)
    else:
        shp = x.shape
        x = x.reshape(shp[0], -1, 2)
        x = torch.bmm(x, u.transpose(-1, -2))
        x = x.reshape(shp)
    return torch.movedim(x, -1, ax)


def _cnot_axes(t: torch.Tensor, ac: int, at: int) -> torch.Tensor:
    """Apply a CNOT permutation on tensor axes ac (control) and at (target)."""
    x0 = t.select(ac, 0)
    x1 = t.select(ac, 1)
    at2 = at if at < ac else at - 1
    x1 = x1.flip(at2)
    return torch.stack([x0, x1], dim=ac)


def sv_apply_cnot(psi: torch.Tensor, c: int, t: int) -> torch.Tensor:
    return _cnot_axes(psi, c + 1, t + 1)


def expz_from_prob(prob: torch.Tensor) -> torch.Tensor:
    """prob (B, 2,...,2) real -> <Z_q> (B, n)."""
    n = prob.dim() - 1
    out = []
    for q in range(n):
        pq = torch.movedim(prob, q + 1, -1).reshape(prob.shape[0], -1, 2).sum(1)
        out.append(pq[:, 0] - pq[:, 1])
    return torch.stack(out, -1)


def sv_expz(psi: torch.Tensor) -> torch.Tensor:
    """<Z_q> for every qubit; returns real (B, n)."""
    return expz_from_prob(psi.real ** 2 + psi.imag ** 2)


# ---------------------------------------------------------------- density matrix (F3.*)
def dm_from_sv(psi: torch.Tensor) -> torch.Tensor:
    b = psi.shape[0]
    n = psi.dim() - 1
    v = psi.reshape(b, -1)
    rho = v.unsqueeze(-1) * v.conj().unsqueeze(-2)
    return rho.reshape((b,) + (2,) * (2 * n))


def dm_zero(batch: int, n: int) -> torch.Tensor:
    return dm_from_sv(sv_zero(batch, n))


def _dm_n(rho: torch.Tensor) -> int:
    return (rho.dim() - 1) // 2


def dm_apply_1q(rho: torch.Tensor, u: torch.Tensor, q: int) -> torch.Tensor:
    """rho -> U rho U^dagger on qubit q. Row axes 1..n, column axes n+1..2n."""
    n = _dm_n(rho)
    rho = sv_apply_1q(rho, u, q)  # rows (axis q+1)
    uc = u.conj()
    # columns: rho_{..., j} -> sum_j' rho_{..., j'} conj(U)_{j j'}
    ax = n + q + 1
    x = torch.movedim(rho, ax, -1)
    if uc.dim() == 2:
        x = x @ uc.transpose(-1, -2)
    else:
        shp = x.shape
        x = torch.bmm(x.reshape(shp[0], -1, 2), uc.transpose(-1, -2)).reshape(shp)
    return torch.movedim(x, -1, ax)


def dm_apply_cnot(rho: torch.Tensor, c: int, t: int) -> torch.Tensor:
    n = _dm_n(rho)
    rho = _cnot_axes(rho, c + 1, t + 1)
    return _cnot_axes(rho, n + c + 1, n + t + 1)


def dm_kraus_1q(rho: torch.Tensor, kraus: list[torch.Tensor], q: int) -> torch.Tensor:
    """F3.0: rho -> sum_k K rho K^dagger on qubit q."""
    out = None
    for k in kraus:
        k = k.to(rho.dtype)
        term = dm_apply_1q(rho, k, q)
        out = term if out is None else out + term
    return out


def _ptrace_replace(rho: torch.Tensor, qubits: list[int]) -> torch.Tensor:
    """Return Tr_Q(rho) (x) I_Q / 2^|Q| with the same axis layout as rho."""
    n = _dm_n(rho)
    x = rho
    for q in qubits:
        d = torch.diagonal(x, dim1=q + 1, dim2=n + q + 1).sum(-1)  # removes both axes
        # re-insert identity/2 on these axes
        eye = torch.eye(2, dtype=x.dtype) / 2
        d = d.unsqueeze(-1).unsqueeze(-1) * eye  # (..., 2, 2) appended at end
        # move appended axes back to positions q+1 and n+q+1
        d = torch.movedim(d, (-2, -1), (q + 1, n + q + 1))
        x = d
    return x


def dm_depol_1q(rho: torch.Tensor, p: float, q: int) -> torch.Tensor:
    """F3.1 Pauli-form depolarizing: (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)
    = (1 - 4p/3) rho + (4p/3) Tr_q(rho) (x) I/2."""
    if p == 0:
        return rho
    a = 4.0 * p / 3.0
    return (1 - a) * rho + a * _ptrace_replace(rho, [q])


def dm_depol_2q(rho: torch.Tensor, p: float, q1: int, q2: int) -> torch.Tensor:
    """F3.2 two-qubit Pauli depolarizing: (1-p) rho + p/15 sum_{P != II} P rho P
    = (1 - 16p/15) rho + (16p/15) Tr_{q1 q2}(rho) (x) I/4."""
    if p == 0:
        return rho
    a = 16.0 * p / 15.0
    return (1 - a) * rho + a * _ptrace_replace(rho, [q1, q2])


def kraus_amp_damp(gamma: float) -> list[torch.Tensor]:
    """F3.3"""
    k0 = torch.tensor([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=torch.complex128)
    k1 = torch.tensor([[0, math.sqrt(gamma)], [0, 0]], dtype=torch.complex128)
    return [k0, k1]


def kraus_phase_damp(lam: float) -> list[torch.Tensor]:
    """F3.4"""
    k0 = torch.tensor([[1, 0], [0, math.sqrt(1 - lam)]], dtype=torch.complex128)
    k1 = torch.tensor([[0, 0], [0, math.sqrt(lam)]], dtype=torch.complex128)
    return [k0, k1]


def thermal_params(t1: float, t2: float, tg: float) -> tuple[float, float]:
    """F3.5: gamma = 1 - exp(-tg/T1); 1/T_phi = 1/T2 - 1/(2 T1); lam = 1 - exp(-2 tg / T_phi).
    Requires T2 <= 2 T1. Off-diagonal decay = sqrt(1-gamma) sqrt(1-lam) = exp(-tg/T2)."""
    if t2 > 2 * t1 + 1e-15:
        raise ValueError("T2 must be <= 2 T1")
    gamma = 1.0 - math.exp(-tg / t1)
    inv_tphi = 1.0 / t2 - 1.0 / (2.0 * t1)
    lam = 1.0 - math.exp(-2.0 * tg * inv_tphi)
    return gamma, lam


def dm_thermal(rho: torch.Tensor, t1: float, t2: float, tg: float, q: int) -> torch.Tensor:
    gamma, lam = thermal_params(t1, t2, tg)
    rho = dm_kraus_1q(rho, kraus_amp_damp(gamma), q)
    if lam > 0:
        rho = dm_kraus_1q(rho, kraus_phase_damp(lam), q)
    return rho


def dm_expz(rho: torch.Tensor) -> torch.Tensor:
    b = rho.shape[0]
    n = _dm_n(rho)
    diag = torch.diagonal(rho.reshape(b, 2 ** n, 2 ** n), dim1=-2, dim2=-1).real
    return expz_from_prob(diag.reshape((b,) + (2,) * n))


def dm_trace(rho: torch.Tensor) -> torch.Tensor:
    b = rho.shape[0]
    n = _dm_n(rho)
    return torch.diagonal(rho.reshape(b, 2 ** n, 2 ** n), dim1=-2, dim2=-1).sum(-1)


# ---------------------------------------------------------------- noise model (F3.6-F3.8)
@dataclass(frozen=True)
class NoiseModel:
    p1: float = 1e-3          # 1q depolarizing (Pauli form)
    p2: float = 1e-2          # 2q depolarizing (Pauli form)
    t1: float = 100e-6        # s
    t2: float = 100e-6        # s
    t_1q: float = 50e-9       # s  [paper:08 Gili]
    t_2q: float = 500e-9      # s  [paper:08 Gili]
    e0: float = 0.02          # P(read 1 | 0)
    e1: float = 0.02          # P(read 0 | 1)
    thermal: bool = True

    def tag(self) -> str:
        return (f"p1={self.p1:g},p2={self.p2:g},T1={self.t1:g},T2={self.t2:g},"
                f"ro=({self.e0:g},{self.e1:g}),thermal={self.thermal}")


IDEAL = None


def readout_affine(expz: torch.Tensor, e0: float, e1: float) -> torch.Tensor:
    """F3.7: <Z>_meas = (1 - e0 - e1) <Z> + (e1 - e0)."""
    return (1.0 - e0 - e1) * expz + (e1 - e0)


def sample_shots(expz: torch.Tensor, shots: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """F3.8: estimate <Z> from `shots` +/-1 samples per qubit (marginal sampling, exact for each <Z_q>)."""
    p0 = ((1.0 + expz) / 2).clamp(0, 1)
    k = torch.distributions.Binomial(total_count=shots, probs=p0).sample() if generator is None else \
        torch.binomial(torch.full_like(p0, float(shots)), p0, generator=generator)
    return 2.0 * k / shots - 1.0


# ---------------------------------------------------------------- fused superoperator path (F3.9, speed)
# A channel E on k qubits is stored as a (4^k x 4^k) matrix S acting on vec(rho_k) with index order
# (row bits, col bits). Applying it touches the tensor once instead of once per Kraus/Pauli term.
def _apply_superop(rho: torch.Tensor, S: torch.Tensor, qubits: list[int]) -> torch.Tensor:
    n = _dm_n(rho)
    k = len(qubits)
    axes = [q + 1 for q in qubits] + [n + q + 1 for q in qubits]
    x = torch.movedim(rho, axes, list(range(rho.dim() - 2 * k, rho.dim())))
    shp = x.shape
    x = x.reshape(-1, 4 ** k) @ S.to(x.dtype).transpose(0, 1)
    x = x.reshape(shp)
    return torch.movedim(x, list(range(rho.dim() - 2 * k, rho.dim())), axes)


def _superop_from_channel(fn, k: int) -> torch.Tensor:
    """Build S column by column by applying the reference channel fn to basis matrices |i><j| (k qubits)."""
    d = 2 ** k
    S = torch.zeros(d * d, d * d, dtype=torch.complex128)
    for i in range(d):
        for j in range(d):
            e = torch.zeros(1, d, d, dtype=torch.complex128)
            e[0, i, j] = 1.0
            out = fn(e.reshape((1,) + (2,) * (2 * k)))
            S[:, i * d + j] = out.reshape(-1)
    return S


_SUPEROP_CACHE: dict = {}


def noise_superops(noise: "NoiseModel") -> tuple[torch.Tensor, torch.Tensor]:
    """S1 = thermal(t_1q) o depol1 on one qubit; S2 = (thermal(t_2q) x thermal(t_2q)) o depol2 on (c, t)."""
    key = noise
    if key in _SUPEROP_CACHE:
        return _SUPEROP_CACHE[key]

    def ch1(r):
        r = dm_depol_1q(r, noise.p1, 0)
        if noise.thermal:
            r = dm_thermal(r, noise.t1, noise.t2, noise.t_1q, 0)
        return r

    def ch2(r):
        r = dm_depol_2q(r, noise.p2, 0, 1)
        if noise.thermal:
            r = dm_thermal(r, noise.t1, noise.t2, noise.t_2q, 0)
            r = dm_thermal(r, noise.t1, noise.t2, noise.t_2q, 1)
        return r

    S1 = _superop_from_channel(ch1, 1)
    S2 = _superop_from_channel(ch2, 2)
    _SUPEROP_CACHE[key] = (S1, S2)
    return S1, S2


def dm_channel_1q(rho, S1, q):
    return _apply_superop(rho, S1, [q])


def dm_channel_2q(rho, S2, c, t):
    return _apply_superop(rho, S2, [c, t])


def set_precision(kind: str):
    """'c128' (tests, reference checks) or 'c64' (experiments; |dz| <= 1e-6 measured in G0-t)."""
    global CDTYPE
    CDTYPE = {"c128": torch.complex128, "c64": torch.complex64}[kind]
