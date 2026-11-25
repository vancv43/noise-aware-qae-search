# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Quantum autoencoder (QAE) denoiser of quantum data.

Data: ground states |psi(h)> of the periodic transverse-field Ising model H = -sum Z_i Z_{i+1} - h sum X_i (n qubits),
corrupted by local Pauli-form depolarizing noise of strength p_in on every qubit (input channel, F9.2).
Model: rho_out = U_E^dag  R_trash( U_E rho_in U_E^dag )  U_E, U_E = layered hardware-efficient ansatz (F1.2), trash = qubits
k..n-1 reset to |0> (F10.2). Hardware noise (F3.6) acts after every gate of encoder and decoder; reset error = bit flip
with probability e_reset after the reset (F10.3). Score: fidelity F = <psi|rho_out|psi> with the clean state (F10.4).
Qubit order: qubit 0 = most significant bit (same as sim.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import sim
from .circuits import CircuitSpec, TOPOLOGIES

H_RANGE = (0.2, 2.0)
SPLIT_N = {"train": 64, "val": 32, "test": 64}
SPLIT_SEED = {"train": 11, "val": 12, "test": 13}


# ---------------------------------------------------------------- data (F9)
def _op(o, i, n):
    m = np.array([[1.0]])
    for j in range(n):
        m = np.kron(m, o if j == i else np.eye(2))
    return m


def tfim_hamiltonian(n: int, h: float) -> np.ndarray:
    X = np.array([[0, 1], [1, 0]], float)
    Z = np.diag([1.0, -1.0])
    H = np.zeros((2 ** n, 2 ** n))
    for i in range(n):
        H -= _op(Z, i, n) @ _op(Z, (i + 1) % n, n)
        H -= h * _op(X, i, n)
    return H


def tfim_ground(n: int, h: float) -> np.ndarray:
    """F9.1 ground state (real, unit norm). Raises if the ground level is degenerate (gap < 1e-9)."""
    w, v = np.linalg.eigh(tfim_hamiltonian(n, h))
    if w[1] - w[0] < 1e-9:
        raise ValueError(f"degenerate ground state at h={h}")
    return v[:, 0]


class TestAccessError(RuntimeError):
    pass


@dataclass
class QSplits:
    """h values per split, drawn U(H_RANGE) with fixed seeds (disjoint with probability 1, asserted).
    Test access raises in modes 'search' and 'pilot' (G0-e)."""
    n: int
    mode: str = "search"
    accessed: list = field(default_factory=list)

    def __post_init__(self):
        self.h = {s: np.sort(np.random.default_rng(SPLIT_SEED[s]).uniform(*H_RANGE, SPLIT_N[s])) for s in SPLIT_N}
        allh = np.concatenate(list(self.h.values()))
        assert len(np.unique(np.round(allh, 12))) == len(allh)

    def states(self, split: str) -> torch.Tensor:
        if split == "test" and self.mode in ("search", "pilot"):
            raise TestAccessError(f"test split requested in mode={self.mode}")
        self.accessed.append(split)
        return torch.tensor(np.stack([tfim_ground(self.n, h) for h in self.h[split]]), dtype=torch.complex128)


def pure_dm(psi: torch.Tensor) -> torch.Tensor:
    """(B, 2^n) -> (B, 2,...,2, 2,...,2)."""
    return sim.dm_from_sv(psi.reshape((psi.shape[0],) + (2,) * int(round(math.log2(psi.shape[1])))))


def input_channel(rho: torch.Tensor, p_in: float) -> torch.Tensor:
    """F9.2 local depolarizing (Pauli form, F3.1) of strength p_in on every qubit."""
    n = sim._dm_n(rho)
    for q in range(n):
        rho = sim.dm_depol_1q(rho, p_in, q)
    return rho


def fidelity(psi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """F10.4 F = <psi| rho |psi>, real, shape (B,)."""
    b, d = psi.shape
    r = rho.reshape(b, d, d)
    p = psi.to(r.dtype)
    return torch.einsum("bi,bij,bj->b", p.conj(), r, p).real


# ---------------------------------------------------------------- QAE (F10)
_K_RESET = None


def _reset_kraus():
    k0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
    k1 = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128)
    return [k0, k1]


def reset_qubit(rho: torch.Tensor, q: int, e_reset: float = 0.0) -> torch.Tensor:
    """F10.2 reset to |0>: K0 = |0><0|, K1 = |0><1|. F10.3 then bit flip with probability e_reset."""
    rho = sim.dm_kraus_1q(rho, _reset_kraus(), q)
    if e_reset > 0:
        x = torch.tensor([[0, 1], [1, 0]], dtype=rho.dtype)
        rho = (1 - e_reset) * rho + e_reset * sim.dm_apply_1q(rho, x, q)
    return rho


PRE_ROT = True


def qae_gates(spec: CircuitSpec) -> list:
    """Encoder gate list: [initial rotation layer (layer 0)] + spec layers shifted to 1..depth. Params (depth+1, |axes|, n)."""
    g = [("ROT", ax, 0, ax_i, q) for ax_i, ax in enumerate(spec.axes) for q in range(spec.n)] if PRE_ROT else []
    off = 1 if PRE_ROT else 0
    for x in spec.gate_list():
        g.append(x if x[0] == "CNOT" else (x[0], x[1], x[2] + off, x[3], x[4]))
    return g


def n_layers_params(spec: CircuitSpec) -> int:
    return spec.depth + (1 if PRE_ROT else 0)


def _apply_gates(rho, gates, params, S1, S2, inverse: bool):
    seq = reversed(gates) if inverse else gates
    for g in seq:
        if g[0] == "CNOT":
            c, t = g[1], g[2]
            rho = sim.dm_apply_cnot(rho, c, t)
            if S2 is not None:
                rho = sim.dm_channel_2q(rho, S2, c, t)
        else:
            _, ax, layer, ax_i, q = g
            th = params[layer, ax_i, q]
            rho = sim.dm_apply_1q(rho, sim.GATE_FN[ax](-th if inverse else th), q)
            if S1 is not None:
                rho = sim.dm_channel_1q(rho, S1, q)
    return rho


def qae_forward(spec: CircuitSpec, k: int, params: torch.Tensor, rho_in: torch.Tensor,
                noise: Optional[sim.NoiseModel] = None) -> torch.Tensor:
    """F10.1 rho_out = U_E^dag R_trash(U_E rho U_E^dag) U_E with per-gate channels when noise is given.
    Reset error e_reset = noise.e0 (same 2% as readout, F3.10) when noise is given, else ideal reset."""
    n = spec.n
    assert 1 <= k < n
    if noise is None:
        return _qae_ideal_unitary(spec, k, params, rho_in)
    S1, S2 = sim.noise_superops(noise)
    gates = qae_gates(spec)
    rho = rho_in.to(sim.CDTYPE)
    rho = _apply_gates(rho, gates, params, S1, S2, inverse=False)
    for q in range(k, n):
        rho = reset_qubit(rho, q, noise.e0 if noise is not None else 0.0)
    rho = _apply_gates(rho, gates, params, S1, S2, inverse=True)
    return rho


def encoder_unitary(spec: CircuitSpec, params: torch.Tensor) -> torch.Tensor:
    """U_E as a (2^n, 2^n) matrix: the gate list applied to every basis vector (columns)."""
    n = spec.n
    psi = torch.eye(2 ** n, dtype=sim.CDTYPE).reshape((2 ** n,) + (2,) * n)
    for g in qae_gates(spec):
        if g[0] == "CNOT":
            psi = sim.sv_apply_cnot(psi, g[1], g[2])
        else:
            _, ax, layer, ax_i, q = g
            psi = sim.sv_apply_1q(psi, sim.GATE_FN[ax](params[layer, ax_i, q]), q)
    return psi.reshape(2 ** n, 2 ** n).transpose(0, 1)


def _qae_ideal_unitary(spec, k, params, rho_in):
    """Ideal path: U rho U^dag with the full unitary, exact reset as a partial trace onto |0><0|, then U^dag . U.
    Equal to the gate-by-gate path (test_unitary_path_equals_gate_path); about 10x faster for n = 6."""
    n = spec.n
    d = 2 ** n
    U = encoder_unitary(spec, params)
    b = rho_in.shape[0]
    r = rho_in.to(sim.CDTYPE).reshape(b, d, d)
    r = U @ r @ U.conj().transpose(0, 1)
    dl, dt = 2 ** k, 2 ** (n - k)
    r4 = r.reshape(b, dl, dt, dl, dt)
    lat = torch.diagonal(r4, dim1=2, dim2=4).sum(-1)  # Tr_trash, (b, dl, dl)
    z = torch.zeros(b, dl, dt, dl, dt, dtype=r.dtype)
    z[:, :, 0, :, 0] = lat
    r = z.reshape(b, d, d)
    r = U.conj().transpose(0, 1) @ r @ U
    return r.reshape((b,) + (2,) * (2 * n))


def n_cnot_qae(spec: CircuitSpec) -> int:
    """F11.2 f2 = CNOTs of encoder + decoder (no cancellation across the reset)."""
    return 2 * spec.n_cnot()


class QAE(torch.nn.Module):
    def __init__(self, spec: CircuitSpec, k: int, seed: int, init_scale: float = 0.1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.spec, self.k = spec, k
        self.theta = torch.nn.Parameter(init_scale * torch.randn((n_layers_params(spec), len(spec.axes), spec.n), generator=g,
                                                                 dtype=torch.float64))

    def forward(self, rho_in, noise=None):
        return qae_forward(self.spec, self.k, self.theta, rho_in, noise)


# ---------------------------------------------------------------- genome v2 (F11.1)
Q_DEPTHS = (1, 2, 3, 4, 5, 6)
Q_ROTS = ("YZ", "XYZ")
Q_LATENT = (1, 2)
Q_LR = (1e-3, 1e-1)
Q_GENES = ("topology", "depth", "rotset", "k", "lr")
D_Q = len(Q_GENES)
BITS_Q = (2, 3, 1, 1, 6)  # BHHO encoding; depth: 8 codes for 6 options (quirk, documented)


def _cat(u, options):
    return options[min(int(u * len(options)), len(options) - 1)]


@dataclass(frozen=True)
class QPhenotype:
    n: int
    topology: str
    depth: int
    rotset: str
    k: int
    lr: float

    @property
    def spec(self) -> CircuitSpec:
        return CircuitSpec(self.n, self.depth, self.topology, self.rotset)

    def key(self) -> tuple:
        return (self.n, self.topology, self.depth, self.rotset, self.k, float(f"{self.lr:.3g}"))


def make_decode_q(n: int):
    def decode_q(g: np.ndarray) -> QPhenotype:
        g = np.clip(np.asarray(g, float), 0, 1)
        assert g.shape == (D_Q,)
        lr = float(math.exp(math.log(Q_LR[0]) + g[4] * (math.log(Q_LR[1]) - math.log(Q_LR[0]))))
        return QPhenotype(n, _cat(g[0], TOPOLOGIES), _cat(g[1], Q_DEPTHS), _cat(g[2], Q_ROTS), _cat(g[3], Q_LATENT), lr)
    return decode_q


def f2_q(ph: QPhenotype) -> int:
    return n_cnot_qae(ph.spec)


def f2_max_q(n: int) -> int:
    return n_cnot_qae(CircuitSpec(n, max(Q_DEPTHS), "all"))


# ---------------------------------------------------------------- training (F10.5)
@dataclass
class QTrainCfg:
    lr_max: float = 1e-2
    epochs: int = 100
    batch: int = 16
    warmup_frac: float = 0.05
    lr_min_frac: float = 0.05
    patience: int = 20
    p_in: float = 0.2
    seed: int = 0


def _lr(step, total, cfg):
    w = max(1, int(round(cfg.warmup_frac * total)))
    if step < w:
        return cfg.lr_max * (step + 1) / w
    lo = cfg.lr_min_frac * cfg.lr_max
    t = (step - w) / max(1, total - w)
    return lo + 0.5 * (cfg.lr_max - lo) * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def eval_fidelity(model: QAE, psi: torch.Tensor, p_in: float, noise=None, batch: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(psi), batch):
        p = psi[i:i + batch]
        out.append(fidelity(p, model(input_channel(pure_dm(p), p_in), noise)))
    return torch.cat(out)


def train_qae(model: QAE, psi_tr: torch.Tensor, psi_val: torch.Tensor, cfg: QTrainCfg) -> dict:
    """Adam on L = 1 - mean F (ideal gates, noisy input). Warmup + cosine, early stop on ideal val fidelity,
    best weights restored. The input channel is deterministic (a CPTP map), so there is no per-epoch resampling."""
    import copy, time
    g = torch.Generator().manual_seed(cfg.seed * 104729 + 3)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_max)
    n = len(psi_tr)
    spe = math.ceil(n / cfg.batch)
    total = spe * cfg.epochs
    rho_tr = input_channel(pure_dm(psi_tr), cfg.p_in)
    step, best, best_state, best_ep, bad, hist = 0, -1.0, None, 0, 0, []
    t0 = time.time()
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            for pg in opt.param_groups:
                pg["lr"] = _lr(step, total, cfg)
            loss = 1 - fidelity(psi_tr[idx], model(rho_tr[idx])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            tot += loss.item() * len(idx)
        vf = float(eval_fidelity(model, psi_val, cfg.p_in).mean())
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_fid": vf, "lr_end": opt.param_groups[0]["lr"]})
        if vf > best + 1e-7:
            best, best_ep, bad, best_state = vf, ep + 1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    tl = [h["train_loss"] for h in hist]
    flags = {"underfit": bool(len(tl) > 1 and tl[-1] > 0.9 * tl[0]), "early_peak": bool(best_ep < 10)}
    return {"best_val_fid": best, "best_epoch": best_ep, "epochs_run": len(hist), "history": hist,
            "seconds": time.time() - t0, "flags": flags}
