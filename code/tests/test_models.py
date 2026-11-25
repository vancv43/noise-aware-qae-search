# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
import math
import numpy as np
import torch
from qmiqcae.circuits import CircuitSpec, run_circuit, edges, count_cnot_after_cancellation, TOPOLOGIES
from qmiqcae.models import build_model, count_params, mlp_hidden_for, HybridAE, QuantumLatent
from qmiqcae.genome import decode, F2_MAX, f2_cnot, D

def test_cnot_count_equals_p_times_edges_and_cancellation_works():
    for n in (4, 6, 8):
        for p in (1, 2, 3, 4):
            for t in TOPOLOGIES:
                s = CircuitSpec(n, p, t)
                assert s.n_cnot() == p * len(edges(t, n))
    assert count_cnot_after_cancellation([("CNOT", 0, 1), ("CNOT", 0, 1)]) == 0
    assert count_cnot_after_cancellation([("CNOT", 0, 1), ("ROT", "RY", 0, 0, 1), ("CNOT", 0, 1)]) == 2
    assert count_cnot_after_cancellation([("CNOT", 0, 1), ("ROT", "RY", 0, 0, 2), ("CNOT", 0, 1)]) == 0
    assert F2_MAX == 112

def test_product_state_without_cnot():
    """F1.6: topology none -> z_i depends only on y_i."""
    spec = CircuitSpec(4, 3, "none", "XYZ")
    th = torch.rand(3, 3, 4, dtype=torch.float64) * 6
    y = torch.rand(1, 4, dtype=torch.float64) * 3
    z0 = run_circuit(spec, y, th)
    y2 = y.clone(); y2[0, 2] += 0.7
    z1 = run_circuit(spec, y2, th)
    d = (z1 - z0).abs()[0]
    assert d[2] > 1e-3 and d[[0, 1, 3]].max() < 1e-12
    spec_r = CircuitSpec(4, 3, "ring", "XYZ")
    d2 = (run_circuit(spec_r, y2, th) - run_circuit(spec_r, y, th)).abs()[0]
    assert d2[[0, 1, 3]].max() > 1e-4  # entangling ring mixes features

def test_param_counts_and_controls():
    spec = CircuitSpec(4, 2, "ring")
    assert spec.n_params == 16
    k = mlp_hidden_for(16, 4); assert k == 4 and mlp_hidden_for(48, 4) == 5 and mlp_hidden_for(48, 8) == 8
    q = build_model("qcae", 4, 8, spec)
    fz = build_model("qcae_frozen", 4, 8, spec)
    ml = build_model("mlp_matched", 4, 8, spec)
    idm = build_model("identity", 4, 8, spec)
    base = count_params(idm)
    assert count_params(q) - base == 16
    assert count_params(fz) - base == 0
    assert count_params(ml) - base == 2 * 4 * k + k + 4
    x = torch.rand(5, 1, 28, 28)
    for m in (q, fz, ml, idm, build_model("lintanh", 4, 8, spec), build_model("cae", 4, 8), build_model("dncnn", 4, 8)):
        out = m(x)
        assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1

def test_decode_covers_all_options_and_bounds():
    rng = np.random.default_rng(0)
    seen = {g: set() for g in ("topology", "depth", "n", "rotset", "width")}
    lrs = []
    for _ in range(4000):
        ph = decode(rng.random(D))
        for g in seen:
            seen[g].add(getattr(ph, g))
        lrs.append(ph.lr)
    assert seen["topology"] == {"ring", "linear", "all", "none"} and seen["depth"] == {1, 2, 3, 4}
    assert seen["n"] == {4, 6, 8} and seen["width"] == {8, 16, 32} and seen["rotset"] == {"YZ", "XYZ"}
    assert abs(decode(np.zeros(D)).lr - 1e-4) < 1e-15 and abs(decode(np.ones(D)).lr - 3e-2) < 1e-12
    assert decode(np.ones(D)).topology == "none" and decode(np.ones(D)).n == 8
