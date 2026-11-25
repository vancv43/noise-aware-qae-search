# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
import numpy as np
import torch
from qmiqcae import sim

def test_readout_affine_matches_monte_carlo():
    rng = np.random.default_rng(1)
    p0 = 0.8; e0, e1 = 0.03, 0.07
    true = rng.random(400000) > p0         # True = |1>
    flip = np.where(true, rng.random(true.size) < e1, rng.random(true.size) < e0)
    meas = true ^ flip
    ez_mc = 1 - 2 * meas.mean()
    ez = sim.readout_affine(torch.tensor([2 * p0 - 1.0]), e0, e1).item()
    assert abs(ez - ez_mc) < 0.005

def test_shot_estimator_unbiased_and_variance():
    ez = torch.full((20000, 1), 0.3, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    est = sim.sample_shots(ez, 1024, g)
    assert abs(est.mean().item() - 0.3) < 0.002
    assert abs(est.var().item() - (1 - 0.09) / 1024) / ((1 - 0.09) / 1024) < 0.05

def test_thermal_offdiag_decay_equals_exp_t_over_t2():
    t1, t2, tg = 80e-6, 60e-6, 500e-9
    g, l = sim.thermal_params(t1, t2, tg)
    assert abs(np.sqrt(1 - g) * np.sqrt(1 - l) - np.exp(-tg / t2)) < 1e-14
