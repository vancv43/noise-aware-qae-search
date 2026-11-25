# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""G0-c: autograd gradient == parameter-shift rule for every rotation parameter (ideal simulation)."""
import math
import numpy as np
import pytest
import torch
from qmiqcae.circuits import CircuitSpec, run_circuit, TOPOLOGIES

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("rotset", ["YZ", "XYZ"])
def test_autograd_equals_parameter_shift(topology, rotset):
    rng = np.random.default_rng(7)
    spec = CircuitSpec(4, 2, topology, rotset)
    y = torch.tensor(rng.uniform(-math.pi, math.pi, (3, 4)))
    th = torch.tensor(rng.uniform(0, 2 * math.pi, (2, len(spec.axes), 4)), requires_grad=True)
    w = torch.tensor(rng.normal(size=(3, 4)))
    L = (run_circuit(spec, y, th) * w).sum()
    g_auto, = torch.autograd.grad(L, th)
    g_psr = torch.zeros_like(th)
    with torch.no_grad():
        for idx in np.ndindex(*th.shape):
            tp = th.detach().clone(); tp[idx] += math.pi / 2
            tm = th.detach().clone(); tm[idx] -= math.pi / 2
            g_psr[idx] = 0.5 * ((run_circuit(spec, y, tp) * w).sum() - (run_circuit(spec, y, tm) * w).sum())
    assert torch.max(torch.abs(g_auto - g_psr)) < 1e-8

def test_gradient_flows_to_encoding():
    spec = CircuitSpec(4, 2, "ring")
    y = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    th = torch.full((2, 2, 4), 0.3, dtype=torch.float64)
    run_circuit(spec, y + 0.4, th).sum().backward()
    assert torch.all(torch.isfinite(y.grad)) and y.grad.abs().sum() > 0
