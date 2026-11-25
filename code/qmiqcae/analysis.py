# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Multi-objective indicators (F4.13-F4.14), mechanism measures (F6.*), statistics (F8.*), Pareto selection (F4.4)."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats

from .circuits import CircuitSpec, final_state
from .genome import F2_MAX
from .search.algorithms import nondominated_mask


# ------------------------------------------------------------------ MO indicators
def normalise(F: np.ndarray, f2_max: float = F2_MAX) -> np.ndarray:
    """(f1, f2 / f2_max). v1 default 112; v2 must pass f2_max_q(n) (72 for n=4, 180 for n=6)."""
    F = np.asarray(F, float)
    return np.column_stack([F[:, 0], F[:, 1] / f2_max])


HV_REF = np.array([1.0, 1.0 + 1.0 / F2_MAX])


def hypervolume(F: np.ndarray, ref: np.ndarray = None, f2_max: float = F2_MAX) -> float:
    """F4.13 exact 2-D hypervolume of the non-dominated subset of normalised F (minimisation).
    Default ref (1, 1 + 1/f2_max)."""
    ref = np.array([1.0, 1.0 + 1.0 / f2_max]) if ref is None else ref
    P = normalise(F, f2_max)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[nondominated_mask(P)]
    P = P[np.argsort(P[:, 0])]
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in P:
        if f2 < prev_f2:
            hv += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def igd_plus(F: np.ndarray, Z: np.ndarray, f2_max: float = F2_MAX) -> float:
    """F4.14 IGD+ of F w.r.t. reference front Z (both normalised inside)."""
    A = normalise(F, f2_max)
    R = normalise(Z, f2_max)
    d = np.sqrt((np.maximum(A[None, :, :] - R[:, None, :], 0) ** 2).sum(-1))
    return float(d.min(1).mean())


def select_pareto_point(F: np.ndarray) -> int:
    """F4.4 pre-registered rule: among non-dominated points with f2 <= median f2 of the front, lowest f1;
    ties -> lowest f2. Returns the index into F."""
    F = np.asarray(F, float)
    nd = np.where(nondominated_mask(F))[0]
    med = np.median(F[nd, 1])
    cand = [i for i in nd if F[i, 1] <= med]
    cand.sort(key=lambda i: (F[i, 0], F[i, 1]))
    return int(cand[0])


# ------------------------------------------------------------------ mechanism measures
def haar_fidelity_pdf(F: np.ndarray, N: int) -> np.ndarray:
    """F6.2 P_Haar(F) = (N-1)(1-F)^(N-2)."""
    return (N - 1) * (1 - F) ** (N - 2)


def expressibility(spec: CircuitSpec, n_pairs: int = 5000, bins: int = 75, seed: int = 0,
                   y: np.ndarray | None = None) -> float:
    """F6.1 KL(P_PQC(F) || P_Haar(F)) [Sim et al. 2019]. Encoding angles fixed to y (default: one random input)."""
    rng = np.random.default_rng(seed)
    yy = rng.uniform(-math.pi, math.pi, spec.n) if y is None else y
    shape = (spec.depth, len(spec.axes), spec.n)
    fids = []
    chunk = 500
    for s in range(0, n_pairs, chunk):
        m = min(chunk, n_pairs - s)
        a = torch.tensor(rng.uniform(0, 2 * math.pi, (m,) + shape))
        b = torch.tensor(rng.uniform(0, 2 * math.pi, (m,) + shape))
        Y = torch.tensor(np.tile(yy, (1, 1)))
        sa = torch.stack([final_state(spec, Y, a[k])[0] for k in range(m)])
        sb = torch.stack([final_state(spec, Y, b[k])[0] for k in range(m)])
        fids.append((torch.abs((sa.conj() * sb).sum(-1)) ** 2).numpy())
    Fv = np.concatenate(fids)
    hist, edges = np.histogram(Fv, bins=bins, range=(0, 1))
    p = hist / hist.sum()
    N = 2 ** spec.n
    # Haar bin probabilities by exact integration: CDF(F) = 1 - (1-F)^(N-1)
    q = (1 - edges[:-1]) ** (N - 1) - (1 - edges[1:]) ** (N - 1)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-300))))


def meyer_wallach(psi: torch.Tensor, n: int) -> torch.Tensor:
    """F6.3 Q = 2 (1 - (1/n) sum_k Tr(rho_k^2)) for pure states psi (B, 2^n)."""
    b = psi.shape[0]
    t = psi.reshape((b,) + (2,) * n)
    pur = []
    for k in range(n):
        x = torch.movedim(t, k + 1, 1).reshape(b, 2, -1)
        rho = x @ x.conj().transpose(-1, -2)
        pur.append(torch.einsum("bij,bji->b", rho, rho).real)
    return 2 * (1 - torch.stack(pur, -1).mean(-1))


def entangling_capability(spec: CircuitSpec, n_samples: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    shape = (spec.depth, len(spec.axes), spec.n)
    vals = []
    for _ in range(0, n_samples, 250):
        y = torch.tensor(rng.uniform(-math.pi, math.pi, (250, spec.n)))
        ps = []
        for k in range(250):
            th = torch.tensor(rng.uniform(0, 2 * math.pi, shape))
            ps.append(final_state(spec, y[k:k + 1], th)[0])
        vals.append(meyer_wallach(torch.stack(ps), spec.n).numpy())
    return float(np.concatenate(vals)[:n_samples].mean())


# ------------------------------------------------------------------ statistics
def a12(x, y) -> float:
    """F8.3 Vargha-Delaney A12 = P(X > Y) + 0.5 P(X = Y)."""
    x = np.asarray(x)[:, None]
    y = np.asarray(y)[None, :]
    return float((x > y).mean() + 0.5 * (x == y).mean())


def holm(pvals: dict) -> dict:
    """F8.2 Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def friedman(table: np.ndarray) -> tuple[float, float]:
    """F8.1 table (blocks x methods)."""
    r = stats.friedmanchisquare(*[table[:, j] for j in range(table.shape[1])])
    return float(r.statistic), float(r.pvalue)


def wilcoxon_vs(ref: np.ndarray, others: dict) -> dict:
    """Paired two-sided Wilcoxon of ref vs each other method, Holm-adjusted, with A12 (ref vs other)."""
    raw = {k: float(stats.wilcoxon(ref, v, alternative="two-sided", zero_method="wilcox").pvalue) for k, v in others.items()}
    adj = holm(raw)
    return {k: {"p": raw[k], "p_holm": adj[k], "a12": a12(ref, others[k])} for k in others}
