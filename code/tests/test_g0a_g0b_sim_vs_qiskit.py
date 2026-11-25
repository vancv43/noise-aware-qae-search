# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""G0-a: statevector vs Qiskit Statevector.  G0-b: density-matrix channels vs Qiskit Aer.

Qiskit is little-endian (qubit 0 = least significant bit); our qubit q maps to Qiskit qubit q with the
state reversed, so we compare <Z_q> values (ordering-free) and full states after a bit-reversal.
"""
import itertools
import math

import numpy as np
import pytest
import torch

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
aer = pytest.importorskip("qiskit_aer")
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoise, depolarizing_error, thermal_relaxation_error, ReadoutError, pauli_error

from qmiqcae import sim
from qmiqcae.circuits import CircuitSpec, TOPOLOGIES, ROTSETS, run_circuit

RNG = np.random.default_rng(20261122)


def qiskit_circuit(spec, y, params):
    qc = QuantumCircuit(spec.n)
    for q in range(spec.n):
        qc.ry(float(y[q]), q)
    for g in spec.gate_list():
        if g[0] == "CNOT":
            qc.cx(g[1], g[2])
        else:
            _, ax, layer, ax_i, q = g
            getattr(qc, ax.lower())(float(params[layer, ax_i, q]), q)
    return qc


def bitrev_state(vec, n):
    idx = np.arange(2 ** n)
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in idx])
    return vec[rev]


def qiskit_expz_sv(qc, n):
    sv = Statevector.from_instruction(qc)
    return np.array([sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], n)).real for q in range(n)])


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("rotset", list(ROTSETS))
def test_g0a_statevector(n, topology, rotset):
    for depth in (1, 3):
        spec = CircuitSpec(n, depth, topology, rotset)
        y = RNG.uniform(-math.pi, math.pi, size=(2, n))
        params = RNG.uniform(0, 2 * math.pi, size=(depth, len(spec.axes), n))
        ours = run_circuit(spec, torch.tensor(y), torch.tensor(params)).numpy()
        from qmiqcae.circuits import final_state
        st = final_state(spec, torch.tensor(y), torch.tensor(params)).numpy()
        for b in range(2):
            qc = qiskit_circuit(spec, y[b], params)
            ref = qiskit_expz_sv(qc, n)
            assert np.max(np.abs(ours[b] - ref)) < 1e-10
            ref_state = bitrev_state(np.asarray(Statevector.from_instruction(qc).data), n)
            assert np.max(np.abs(st[b] - ref_state)) < 1e-10


def aer_noise_model(nm: sim.NoiseModel):
    m = AerNoise()
    # Pauli-form depolarizing via pauli_error, exactly our parametrization
    e1 = pauli_error([("I", 1 - nm.p1), ("X", nm.p1 / 3), ("Y", nm.p1 / 3), ("Z", nm.p1 / 3)])
    paulis2 = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]
    e2 = pauli_error([(p, (1 - nm.p2) if p == "II" else nm.p2 / 15) for p in paulis2])
    if nm.thermal:
        e1 = e1.compose(thermal_relaxation_error(nm.t1, nm.t2, nm.t_1q))
        tr2 = thermal_relaxation_error(nm.t1, nm.t2, nm.t_2q)
        e2 = e2.compose(tr2.expand(tr2))
    m.add_all_qubit_quantum_error(e1, ["ry", "rz", "rx"])
    m.add_all_qubit_quantum_error(e2, ["cx"])
    return m


def aer_noisy_expz(qc, nm, n):
    qc2 = qc.copy()
    qc2.save_density_matrix()
    simr = AerSimulator(method="density_matrix", noise_model=aer_noise_model(nm))
    res = simr.run(qc2, optimization_level=0).result()
    rho = DensityMatrix(res.data(0)["density_matrix"])
    ez = np.array([rho.expectation_value(SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], n)).real for q in range(n)])
    return (1 - nm.e0 - nm.e1) * ez + (nm.e1 - nm.e0)


@pytest.mark.parametrize("n", [2, 4, 6])
@pytest.mark.parametrize("thermal", [False, True])
def test_g0b_density_matrix_vs_aer(n, thermal):
    nm = sim.NoiseModel(p1=0.01, p2=0.05, t1=60e-6, t2=40e-6, e0=0.03, e1=0.05, thermal=thermal)
    for topology in TOPOLOGIES:
        spec = CircuitSpec(n, 2, topology, "XYZ")
        y = RNG.uniform(-math.pi, math.pi, size=(1, n))
        params = RNG.uniform(0, 2 * math.pi, size=(2, 3, n))
        ours = run_circuit(spec, torch.tensor(y), torch.tensor(params), noise=nm).numpy()[0]
        qc = qiskit_circuit(spec, y[0], params)
        ref = aer_noisy_expz(qc, nm, n)
        assert np.max(np.abs(ours - ref)) < 1e-6, (topology, ours, ref)
        ideal = run_circuit(spec, torch.tensor(y), torch.tensor(params)).numpy()[0]
        assert np.max(np.abs(ours - ideal)) > 1e-3  # the noise really acts (guards a silent no-op)


def test_g0b_depol_formula_equals_kraus():
    """Partial-trace form of depolarizing == explicit Pauli Kraus sum."""
    n = 3
    psi = torch.tensor(RNG.normal(size=8) + 1j * RNG.normal(size=8))
    psi = (psi / psi.norm()).reshape(1, 2, 2, 2)
    rho = sim.dm_from_sv(psi)
    p = 0.13
    a = sim.dm_depol_1q(rho, p, 1)
    kr = [math.sqrt(1 - p) * sim.I2, math.sqrt(p / 3) * sim.PX, math.sqrt(p / 3) * sim.PY, math.sqrt(p / 3) * sim.PZ]
    b = sim.dm_kraus_1q(rho, kr, 1)
    assert torch.max(torch.abs(a - b)) < 1e-12
    # 2q
    P = [sim.I2, sim.PX, sim.PY, sim.PZ]
    a2 = sim.dm_depol_2q(rho, p, 0, 2)
    b2 = 0
    for i, j in itertools.product(range(4), repeat=2):
        w = (1 - p) if (i == 0 and j == 0) else p / 15
        r = sim.dm_apply_1q(sim.dm_apply_1q(rho, P[i], 0), P[j], 2)
        b2 = b2 + w * r
    assert torch.max(torch.abs(a2 - b2)) < 1e-12
    assert abs(sim.dm_trace(a2)[0].real - 1) < 1e-12


def test_qiskit_depolarizing_mapping():
    """Qiskit depolarizing_error(lam, d) == our Pauli form with p = lam (d^2-1)/d^2."""
    from qiskit.quantum_info import SuperOp
    for d_q, lam in [(1, 0.08), (2, 0.08)]:
        e = depolarizing_error(lam, d_q)
        d = 2 ** d_q
        p = lam * (d * d - 1) / (d * d)
        paulis = ["".join(x) for x in itertools.product("IXYZ", repeat=d_q)]
        ours = pauli_error([(s, (1 - p) if set(s) == {"I"} else p / (d * d - 1)) for s in paulis])
        assert np.allclose(SuperOp(e).data, SuperOp(ours).data, atol=1e-12)


def test_superop_path_equals_kraus_path():
    """Fused superoperator (F3.9) == sequential depol + thermal Kraus channels."""
    nm = sim.NoiseModel(p1=0.02, p2=0.07, t1=50e-6, t2=70e-6)
    S1, S2 = sim.noise_superops(nm)
    psi = torch.tensor(RNG.normal(size=16) + 1j * RNG.normal(size=16))
    rho = sim.dm_from_sv((psi / psi.norm()).reshape(1, 2, 2, 2, 2))
    a = sim.dm_channel_1q(rho, S1, 2)
    b = sim.dm_thermal(sim.dm_depol_1q(rho, nm.p1, 2), nm.t1, nm.t2, nm.t_1q, 2)
    assert torch.max(torch.abs(a - b)) < 1e-13
    a2 = sim.dm_channel_2q(rho, S2, 3, 1)
    b2 = sim.dm_depol_2q(rho, nm.p2, 3, 1)
    b2 = sim.dm_thermal(sim.dm_thermal(b2, nm.t1, nm.t2, nm.t_2q, 3), nm.t1, nm.t2, nm.t_2q, 1)
    assert torch.max(torch.abs(a2 - b2)) < 1e-13


def test_c64_matches_c128_noisy():
    spec = CircuitSpec(6, 3, "all", "XYZ")
    y = torch.tensor(RNG.uniform(-3, 3, (4, 6)))
    th = torch.tensor(RNG.uniform(0, 6, (3, 3, 6)))
    z128 = run_circuit(spec, y, th, noise=sim.NoiseModel())
    sim.set_precision("c64")
    try:
        z64 = run_circuit(spec, y.float(), th.float(), noise=sim.NoiseModel())
    finally:
        sim.set_precision("c128")
    assert torch.max(torch.abs(z128 - z64.double())) < 1e-5
