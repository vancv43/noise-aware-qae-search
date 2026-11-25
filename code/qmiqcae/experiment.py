# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Glue: search-time evaluation function and one search run per (method, seed). Reads train/val only."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from . import sim
from .data import SplitStore
from .genome import D, F2_MAX, Phenotype
from .models import build_model, count_params
from .problem import Problem
from .search import algorithms as A
from .train import TrainCfg, train_model, evaluate

METHODS_MO = ("mopso", "nsga2", "moead", "random")
METHODS_SO = ("pso", "hho", "bhho")
LAMBDA_SO = 0.1


def make_search_evaluator(store: SplitStore, search_noise: Optional[sim.NoiseModel], epochs: int, sigma: float,
                          run_seed: int, val_noise_seed: int = 1001, n_val_noisy_eval: Optional[int] = None):
    """Returns evaluate_fn(phenotype, eval_index) -> dict with f1 and diagnostics.

    Training: ideal statevector, E_s epochs, full warmup+cosine schedule compressed into E_s (F2.2).
    f1 = 1 - SSIM_val under search_noise (None = ideal arm).
    """
    assert store.mode in ("search", "pilot")
    x_tr, _ = store.get("train")
    v_noisy, v_clean, _ = store.fixed_noisy("val", sigma, val_noise_seed)
    if n_val_noisy_eval is not None:
        v_noisy, v_clean = v_noisy[:n_val_noisy_eval], v_clean[:n_val_noisy_eval]

    def fn(ph: Phenotype, idx: int) -> dict:
        seed = run_seed * 100003 + idx
        model = build_model("qcae", ph.n, ph.width, ph.spec, seed=seed)
        cfg = TrainCfg(lr_max=ph.lr, weight_decay=ph.wd, sigma=sigma, seed=seed, max_epochs=epochs,
                       patience=epochs + 1)  # no early stop inside the short search budget
        model, info = train_model(model, x_tr, v_noisy, v_clean, cfg)
        t0 = time.time()
        ev = evaluate(model, v_noisy, v_clean, noise=search_noise)
        return {"f1": 1.0 - ev["ssim"], "val_ssim_ideal": info["best_val_ssim"], "val_ssim_search": ev["ssim"],
                "best_epoch": info["best_epoch"], "train_seconds": info["seconds"], "eval_seconds": time.time() - t0,
                "flags": info["flags"], "n_params": count_params(model)}

    return fn


def run_search(method: str, problem: Problem, seed: int, pop: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    if method == "mopso":
        out = A.mopso(problem, D, rng, n_particles=pop, budget_hint=problem.budget)
    elif method == "nsga2":
        out = A.nsga2(problem, D, seed, pop_size=pop)
    elif method == "moead":
        out = A.moead(problem, D, seed, pop_size=pop)
    elif method == "random":
        out = A.random_search(problem, D, rng)
    elif method in METHODS_SO:
        so = A.Scalarized(problem, LAMBDA_SO, F2_MAX)
        if method == "pso":
            out = A.pso(so, D, rng, n=pop, budget_hint=problem.budget)
        elif method == "hho":
            out = A.hho(so, D, rng, n=pop, budget_hint=problem.budget)
        else:
            out = A.bhho(so, rng, n=pop, budget_hint=problem.budget)
    else:
        raise ValueError(method)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    return {"method": method, "seed": seed, "problem": problem.summary(), "algo_out": out}
