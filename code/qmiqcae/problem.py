# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Evaluation wrapper shared by every search method: cache, exact unique-evaluation budget, failure log.

An evaluation trains a QCAE for E_s epochs (ideal simulation), then scores f1 = 1 - SSIM_val under the
search-condition model (noise model for NA arms, ideal for the ideal arm), and f2 = CNOT count.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np

from .genome import decode, f2_cnot, Phenotype


class BudgetExhausted(Exception):
    pass


class Problem:
    def __init__(self, evaluate_fn: Callable[[Phenotype, int], dict], budget: int, max_proposals_factor: int = 20,
                 log_path: Optional[str] = None, decode_fn: Callable = decode, f2_fn: Callable = f2_cnot):
        """evaluate_fn(phenotype, eval_index) -> {'f1': float, ...}. Must not read the test split."""
        self.evaluate_fn = evaluate_fn
        self.budget = budget
        self.max_proposals = max_proposals_factor * budget
        self.cache: dict = {}
        self.records: list = []
        self.n_unique = 0
        self.n_proposals = 0
        self.n_cache_hits = 0
        self.n_failures = 0
        self.log_path = log_path
        self.decode_fn, self.f2_fn = decode_fn, f2_fn  # v2 (QAE) passes its own genome decoder and f2

    @property
    def exhausted(self) -> bool:
        return self.n_unique >= self.budget or self.n_proposals >= self.max_proposals

    def __call__(self, g: np.ndarray) -> tuple[float, float]:
        """Returns (f1, f2). Raises BudgetExhausted when no budget is left for a NEW phenotype."""
        ph = self.decode_fn(g)
        key = ph.key()
        self.n_proposals += 1
        if key in self.cache:
            self.n_cache_hits += 1
            r = self.cache[key]
            return r["f1"], r["f2"]
        if self.exhausted:
            raise BudgetExhausted
        t0 = time.time()
        idx = self.n_unique
        self.n_unique += 1
        try:
            out = self.evaluate_fn(ph, idx)
            f1 = float(out["f1"])
            if not np.isfinite(f1):
                raise ValueError("non-finite f1")
            status = "ok"
        except Exception as e:  # failed trial is recorded and the search continues (worst f1)
            out = {"error": repr(e), "trace": traceback.format_exc()[-2000:]}
            f1 = 1.0
            status = "failed"
            self.n_failures += 1
        rec = {"eval": idx, "genome": [float(x) for x in np.clip(g, 0, 1)], "phenotype": asdict(ph),
               "f1": f1, "f2": int(self.f2_fn(ph)), "status": status, "seconds": time.time() - t0,
               **{k: v for k, v in out.items() if k not in ("f1",)}}
        self.cache[key] = rec
        self.records.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
        return rec["f1"], rec["f2"]

    def objectives(self) -> np.ndarray:
        return np.array([[r["f1"], r["f2"]] for r in self.records], dtype=float)

    def summary(self) -> dict:
        return {"budget": self.budget, "unique": self.n_unique, "proposals": self.n_proposals,
                "cache_hits": self.n_cache_hits, "failures": self.n_failures}
