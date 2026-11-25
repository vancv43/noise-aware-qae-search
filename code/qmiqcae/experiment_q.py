# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""v2 glue: search-time evaluator for the QAE denoiser and one search run. Reads train/val only."""
from __future__ import annotations
import time
from typing import Optional
import numpy as np, torch
from . import sim
from . import qae as Q
from .search import algorithms as A

LAMBDA_SO = 0.1
MAIN_NOISE = sim.NoiseModel()  # F3.10, frozen


def make_search_evaluator_q(splits: Q.QSplits, search_noise: Optional[sim.NoiseModel], epochs: int, run_seed: int,
                            p_in: float = 0.2):
    """f1 = 1 - mean val fidelity under search_noise (None = ideal arm); training uses ideal gates."""
    assert splits.mode in ("search", "pilot")
    tr, va = splits.states("train"), splits.states("val")

    def fn(ph: Q.QPhenotype, idx: int) -> dict:
        seed = run_seed * 100003 + idx
        m = Q.QAE(ph.spec, ph.k, seed=seed)
        info = Q.train_qae(m, tr, va, Q.QTrainCfg(lr_max=ph.lr, epochs=epochs, patience=epochs + 1, p_in=p_in, seed=seed))
        t0 = time.time()
        fn_ = float(Q.eval_fidelity(m, va, p_in, noise=MAIN_NOISE).mean())  # always recorded (C2 uses it for both arms)
        fs = fn_ if search_noise is not None else info["best_val_fid"]
        return {"f1": 1 - fs, "val_fid_ideal": info["best_val_fid"], "val_fid_noisy": fn_, "best_epoch": info["best_epoch"],
                "train_seconds": info["seconds"], "eval_seconds": time.time() - t0, "flags": info["flags"],
                "search_condition": "noisy" if search_noise is not None else "ideal"}
    return fn


def run_search_q(method: str, problem, seed: int, n: int, pop: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    D = Q.D_Q
    if method == "mopso":
        out = A.mopso(problem, D, rng, n_particles=pop, budget_hint=problem.budget)
    elif method == "nsga2":
        out = A.nsga2(problem, D, seed, pop_size=pop)
    elif method == "moead":
        out = A.moead(problem, D, seed, pop_size=pop)
    elif method == "random":
        out = A.random_search(problem, D, rng)
    else:
        so = A.Scalarized(problem, LAMBDA_SO, Q.f2_max_q(n))
        if method == "pso":
            out = A.pso(so, D, rng, n=pop, budget_hint=problem.budget)
        elif method == "hho":
            out = A.hho(so, D, rng, n=pop, budget_hint=problem.budget)
        elif method == "bhho":
            out = A.bhho(so, rng, n=pop, budget_hint=problem.budget, bits=Q.BITS_Q)
        else:
            raise ValueError(method)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    return {"method": method, "seed": seed, "problem": problem.summary(), "algo_out": out}
