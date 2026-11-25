# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Search methods (F4.6-F4.12). All work on genomes in [0,1]^D and stop when the evaluator raises BudgetExhausted.

Evaluator protocol: ev(x: np.ndarray) -> tuple/array of objectives (minimised); raises BudgetExhausted.
For the QCAE problem the evaluator is qmiqcae.problem.Problem (f1, f2). For G0-d benchmarks it is BenchEvaluator.
Single-objective methods receive a scalarising wrapper (F4.5).

Baseline sources (compared line by line with the reference code):
  HHO  - Heidari et al. 2019, official HHO.m (github aliasgharheidaricom/..., commit e5d854f)
  BHHO - Too et al. 2019, jBHHO.m (github JingweiToo/Binary-Harris-Hawk-..., commit 6c20341)
  NSGA-II, MOEA/D - pymoo 0.6.1.5
"""
from __future__ import annotations

import math

import numpy as np

from ..problem import BudgetExhausted


# ------------------------------------------------------------------ shared MO utilities
def dominates(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    m = np.ones(n, bool)
    for i in range(n):
        if not m[i]:
            continue
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                m[i] = False
                break
    # drop exact duplicates in objective space (keep first)
    seen = set()
    for i in range(n):
        if m[i]:
            k = tuple(F[i])
            if k in seen:
                m[i] = False
            seen.add(k)
    return m


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """F4.9 NSGA-II crowding distance on min-max normalised objectives; boundary points = inf."""
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for k in range(m):
        o = np.argsort(F[:, k], kind="stable")
        lo, hi = F[o[0], k], F[o[-1], k]
        d[o[0]] = d[o[-1]] = np.inf
        if hi - lo <= 0:
            continue
        d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / (hi - lo)
    return d


class BenchEvaluator:
    """Counting evaluator for data-free benchmarks."""

    def __init__(self, func, budget: int):
        self.func = func
        self.budget = budget
        self.n = 0
        self.X, self.F = [], []

    @property
    def exhausted(self):
        return self.n >= self.budget

    def __call__(self, x):
        if self.n >= self.budget:
            raise BudgetExhausted
        self.n += 1
        f = np.atleast_1d(np.asarray(self.func(np.clip(x, 0, 1)), float))
        self.X.append(np.array(x, float))
        self.F.append(f)
        return tuple(f)


class Scalarized:
    """Wrap a MO evaluator for single-objective methods: f = f1 + lam*f2/f2_max (F4.5)."""

    def __init__(self, ev, lam: float, f2_max: float):
        self.ev, self.lam, self.f2_max = ev, lam, f2_max

    @property
    def exhausted(self):
        return self.ev.exhausted

    def __call__(self, x):
        f = self.ev(x)
        return float(f[0] + self.lam * f[1] / self.f2_max)


# ------------------------------------------------------------------ NA-MOPSO (proposed)
def mopso(ev, dim: int, rng: np.random.Generator, n_particles: int = 12, archive_size: int = 20,
          w_start: float = 0.9, w_end: float = 0.4, c1: float = 1.5, c2: float = 1.5, vmax: float = 0.5,
          mut_power: float = 1.5, budget_hint: int = 100):
    """F4.6-F4.10. Schedules use tau = evaluations_used/budget_hint."""
    X = rng.random((n_particles, dim))
    V = np.zeros_like(X)
    P, PF = X.copy(), [None] * n_particles
    AX, AF = [], []
    used = 0

    def update_archive(x, f):
        nonlocal AX, AF
        f = np.asarray(f, float)
        for af in AF:
            if dominates(af, f) or np.array_equal(af, f):
                return
        keep = [i for i, af in enumerate(AF) if not dominates(f, af)]
        AX = [AX[i] for i in keep] + [x.copy()]
        AF = [AF[i] for i in keep] + [f]
        while len(AF) > archive_size:  # F4.8 truncate the most crowded
            cd = crowding_distance(np.array(AF))
            j = int(np.argmin(cd))
            del AX[j]
            del AF[j]

    try:
        for i in range(n_particles):
            f = np.asarray(ev(X[i]), float)
            used += 1
            PF[i] = f
            update_archive(X[i], f)
        while True:
            for i in range(n_particles):
                tau = min(1.0, used / budget_hint)
                w = w_start - (w_start - w_end) * tau
                # F4.7 leader: binary tournament on crowding distance in the archive
                AFa = np.array(AF)
                cd = crowding_distance(AFa)
                a, b = rng.integers(len(AF)), rng.integers(len(AF))
                lead = AX[a] if cd[a] >= cd[b] else AX[b]
                r1, r2 = rng.random(dim), rng.random(dim)
                V[i] = w * V[i] + c1 * r1 * (P[i] - X[i]) + c2 * r2 * (lead - X[i])
                V[i] = np.clip(V[i], -vmax, vmax)
                X[i] = X[i] + V[i]
                out = (X[i] < 0) | (X[i] > 1)
                X[i] = np.clip(X[i], 0, 1)
                V[i][out] = 0.0
                # F4.10 turbulence
                if rng.random() < (1 - tau) ** (1 / mut_power):
                    d = rng.integers(dim)
                    X[i, d] = rng.random()
                f = np.asarray(ev(X[i]), float)
                used += 1
                if dominates(f, PF[i]) or (not dominates(PF[i], f) and rng.random() < 0.5):  # F4.11 pbest
                    P[i], PF[i] = X[i].copy(), f
                update_archive(X[i], f)
    except BudgetExhausted:
        pass
    return {"archive_X": np.array(AX), "archive_F": np.array(AF)}


# ------------------------------------------------------------------ pymoo baselines
def _pymoo_loop(algorithm, ev, dim, seed):
    from pymoo.core.problem import Problem as PProblem
    from pymoo.core.evaluator import Evaluator

    class _P(PProblem):
        def __init__(self, n_obj):
            super().__init__(n_var=dim, n_obj=n_obj, xl=0.0, xu=1.0)

    probe_obj = 2
    prob = _P(probe_obj)
    algorithm.setup(prob, seed=seed, termination=("n_gen", 10 ** 9), verbose=False)
    try:
        while True:
            pop = algorithm.ask()
            from pymoo.core.individual import Individual
            if isinstance(pop, Individual):  # MOEA/D is steady-state: ask() returns one offspring
                pop.set("F", np.asarray(ev(pop.get("X")), float))
            else:
                Fs = [np.asarray(ev(x), float) for x in pop.get("X")]
                pop.set("F", np.array(Fs))
            algorithm.tell(infills=pop)
    except BudgetExhausted:
        pass
    return {}


def nsga2(ev, dim, seed, pop_size=12):
    """pymoo NSGA2 defaults: SBX(eta=15, prob=0.9), PM(eta=20), random init, duplicate elimination."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    return _pymoo_loop(NSGA2(pop_size=pop_size), ev, dim, seed)


def moead(ev, dim, seed, pop_size=12):
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions
    rd = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
    return _pymoo_loop(MOEAD(rd, n_neighbors=min(5, pop_size)), ev, dim, seed)


def random_search(ev, dim, rng):
    try:
        while True:
            ev(rng.random(dim))
    except BudgetExhausted:
        pass
    return {}


# ------------------------------------------------------------------ single-objective baselines
def pso(fobj, dim, rng, n=12, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, vmax=0.5, budget_hint=100):
    """Global-best PSO, SNCS Eqs. 1-2 with linearly decreasing inertia."""
    X = rng.random((n, dim))
    V = np.zeros_like(X)
    P = X.copy()
    PF = np.full(n, np.inf)
    g, gf = X[0].copy(), np.inf
    used = 0
    try:
        for i in range(n):
            PF[i] = fobj(X[i]); used += 1
            if PF[i] < gf:
                gf, g = PF[i], X[i].copy()
        while True:
            for i in range(n):
                w = w_start - (w_start - w_end) * min(1.0, used / budget_hint)
                V[i] = np.clip(w * V[i] + c1 * rng.random(dim) * (P[i] - X[i]) + c2 * rng.random(dim) * (g - X[i]),
                               -vmax, vmax)
                X[i] = np.clip(X[i] + V[i], 0, 1)
                f = fobj(X[i]); used += 1
                if f < PF[i]:
                    PF[i], P[i] = f, X[i].copy()
                if f < gf:
                    gf, g = f, X[i].copy()
    except BudgetExhausted:
        pass
    return {"best_x": g, "best_f": gf}


def levy(dim, rng, beta=1.5, scale=1.0):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return scale * u / np.abs(v) ** (1 / beta)


def hho(fobj, dim, rng, n=12, T=None, budget_hint=100):
    """Port of official HHO.m (Heidari 2019) incl. its quirks: q<0.5 branch = perching on random hawk
    (paper labels swapped), Levy without the 0.01 factor, X1/X2 evaluated unclipped (our decode clips),
    fobj(X_i) re-evaluated in dives (cache hit in our Problem). lb=0, ub=1. T derived from budget_hint."""
    lb, ub = 0.0, 1.0
    if T is None:
        T = max(1, budget_hint // n)
    X = rng.random((n, dim)) * (ub - lb) + lb
    rabbit, rabbit_f = np.zeros(dim), np.inf
    t = 0
    try:
        while True:  # runs past T until the budget is used; E1 then stays at its floor via max(0, .)
            for i in range(n):
                X[i] = np.clip(X[i], lb, ub)
                f = fobj(X[i])
                if f < rabbit_f:
                    rabbit_f, rabbit = f, X[i].copy()
            E1 = 2 * max(0.0, 1 - t / T)
            Xm = X.mean(0)
            for i in range(n):
                E = E1 * (2 * rng.random() - 1)
                if abs(E) >= 1:
                    q = rng.random()
                    Xr = X[int(math.floor(n * rng.random()))]
                    if q < 0.5:
                        X[i] = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * X[i])
                    else:
                        X[i] = (rabbit - X.mean(0)) - rng.random() * ((ub - lb) * rng.random() + lb)
                else:
                    r = rng.random()
                    if r >= 0.5 and abs(E) < 0.5:
                        X[i] = rabbit - E * np.abs(rabbit - X[i])
                    if r >= 0.5 and abs(E) >= 0.5:
                        J = 2 * (1 - rng.random())
                        X[i] = (rabbit - X[i]) - E * np.abs(J * rabbit - X[i])
                    if r < 0.5 and abs(E) >= 0.5:
                        J = 2 * (1 - rng.random())
                        X1 = rabbit - E * np.abs(J * rabbit - X[i])
                        if fobj(X1) < fobj(X[i]):
                            X[i] = X1
                        else:
                            X2 = X1 + rng.random(dim) * levy(dim, rng)
                            if fobj(X2) < fobj(X[i]):
                                X[i] = X2
                    if r < 0.5 and abs(E) < 0.5:
                        J = 2 * (1 - rng.random())
                        X1 = rabbit - E * np.abs(J * rabbit - X.mean(0))
                        if fobj(X1) < fobj(X[i]):
                            X[i] = X1
                        else:
                            X2 = X1 + rng.random(dim) * levy(dim, rng)
                            if fobj(X2) < fobj(X[i]):
                                X[i] = X2
            t += 1
    except BudgetExhausted:
        pass
    return {"best_x": rabbit, "best_f": rabbit_f}


# -------- BHHO on a bit-encoded genome
BITS = (2, 2, 2, 1, 6, 6, 2)  # topology, depth, n, rotset, lr, wd, width -> 21 bits


def bits_to_genome(b: np.ndarray, bits=BITS) -> np.ndarray:
    """Gene with k bits -> integer v -> u = (v + 0.5)/2^k (bin centre) -> decoded by genome.decode."""
    out, pos = [], 0
    for k in bits:
        v = 0
        for j in range(k):
            v = (v << 1) | int(b[pos + j])
        out.append((v + 0.5) / 2 ** k)
        pos += k
    return np.array(out)


def bhho(fobj_genome, rng, n=12, T=None, budget_hint=100, bits=BITS):
    """Port of jBHHO.m (Too et al. 2019): binary hawks, sigmoid transfer S = 1/(1+e^-Xn), random init P(1)=0.5,
    soft/hard besiege without greedy step, dives build Y then Z = Y + rand*LF with greedy accept; LF has 0.01."""
    dim = int(sum(bits))
    lb, ub = 0.0, 1.0
    if T is None:
        T = max(1, budget_hint // n)
    f = lambda b: fobj_genome(bits_to_genome(b, bits))
    S = lambda x: 1.0 / (1.0 + np.exp(-x))
    X = (rng.random((n, dim)) > 0.5).astype(float)
    fit = np.zeros(n)
    fitR, Xrb = np.inf, X[0].copy()
    t = 1
    try:
        while True:
            for i in range(n):
                fit[i] = f(X[i])
                if fit[i] < fitR:
                    fitR, Xrb = fit[i], X[i].copy()
            Xmu = X.mean(0)
            for i in range(n):
                E = 2 * (2 * rng.random() - 1) * max(0.0, 1 - t / T)
                if abs(E) >= 1:
                    q = rng.random()
                    if q >= 0.5:
                        k = rng.integers(n)
                        r1, r2 = rng.random(), rng.random()
                        Xn = X[k] - r1 * np.abs(X[k] - 2 * r2 * X[i])
                    else:
                        r3, r4 = rng.random(), rng.random()
                        Xn = (Xrb - Xmu) - r3 * (lb + r4 * (ub - lb))
                    X[i] = (rng.random(dim) < S(Xn)).astype(float)
                else:
                    J = 2 * (1 - rng.random())
                    r = rng.random()
                    if r >= 0.5 and abs(E) >= 0.5:
                        Xn = (Xrb - X[i]) - E * np.abs(J * Xrb - X[i])
                        X[i] = (rng.random(dim) < S(Xn)).astype(float)
                    elif r >= 0.5 and abs(E) < 0.5:
                        Xn = Xrb - E * np.abs(Xrb - X[i])
                        X[i] = (rng.random(dim) < S(Xn)).astype(float)
                    else:
                        LF = levy(dim, rng, scale=0.01)
                        base = X[i] if abs(E) >= 0.5 else Xmu
                        Yn = Xrb - E * np.abs(J * Xrb - base)
                        Y = (rng.random(dim) < S(Yn)).astype(float)
                        Zn = Y + rng.random(dim) * LF
                        Z = (rng.random(dim) < S(Zn)).astype(float)
                        fy, fz = f(Y), f(Z)
                        if fy <= fit[i]:
                            fit[i], X[i] = fy, Y
                        if fz <= fit[i]:
                            fit[i], X[i] = fz, Z
            t += 1
    except BudgetExhausted:
        pass
    return {"best_bits": Xrb, "best_f": fitR}
