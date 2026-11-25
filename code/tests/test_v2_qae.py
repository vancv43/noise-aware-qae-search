# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""G0 v2: QAE simulator vs Qiskit (ideal: quantum_info DensityMatrix with reset; noisy: Aer density matrix with reset),
fidelity, inverse decoder, input channel, data guard, gradients, genome decode, budget."""
import itertools, math
import numpy as np, pytest, torch
from qmiqcae import sim
from qmiqcae.circuits import CircuitSpec, TOPOLOGIES
from qmiqcae import qae as Q

RNG = np.random.default_rng(7)
qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoise, pauli_error, thermal_relaxation_error


def bitrev(n):
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(2 ** n)])


def qc_qae(spec, k, params, psi_ours, with_reset=True, aer_init=False):
    n = spec.n
    qc = QuantumCircuit(n)
    if aer_init:  # noiseless state load (initialize would be compiled into noisy gates by Aer)
        import qiskit_aer.library  # noqa: registers set_density_matrix
        qc.set_density_matrix(DensityMatrix(Statevector(psi_ours[bitrev(n)])))
    else:
        qc.initialize(Statevector(psi_ours[bitrev(n)]), range(n))
    gl = Q.qae_gates(spec)
    def emit(g, inv):
        if g[0] == "CNOT":
            qc.cx(g[1], g[2])
        else:
            _, ax, layer, ax_i, q = g
            th = float(params[layer, ax_i, q])
            getattr(qc, ax.lower())(-th if inv else th, q)
    for g in gl: emit(g, False)
    if with_reset:
        for q in range(k, n): qc.reset(q)
    for g in reversed(gl): emit(g, True)
    return qc


def dm_to_ours(rho_q, n):
    r = bitrev(n)
    return rho_q[np.ix_(r, r)]


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_qae_ideal_vs_qiskit(n, topology):
    for rotset, depth, k in (("YZ", 2, 1), ("XYZ", 3, 2)):
        spec = CircuitSpec(n, depth, topology, rotset)
        th = RNG.uniform(0, 2 * math.pi, (depth + 1, len(spec.axes), n))
        psi = Q.tfim_ground(n, 0.7)
        ours = Q.qae_forward(spec, k, torch.tensor(th), Q.pure_dm(torch.tensor(psi[None], dtype=torch.complex128)))
        ours = ours.reshape(2 ** n, 2 ** n).numpy()
        qc = qc_qae(spec, k, th, psi)
        ref = dm_to_ours(np.asarray(DensityMatrix(qc).data), n)
        assert np.max(np.abs(ours - ref)) < 1e-10


def test_qae_noisy_vs_aer():
    nm = sim.NoiseModel()
    n = 4
    m = AerNoise()
    e1 = pauli_error([("I", 1 - nm.p1), ("X", nm.p1 / 3), ("Y", nm.p1 / 3), ("Z", nm.p1 / 3)])
    e2 = pauli_error([("".join(p), (1 - nm.p2) if "".join(p) == "II" else nm.p2 / 15) for p in itertools.product("IXYZ", repeat=2)])
    e1 = e1.compose(thermal_relaxation_error(nm.t1, nm.t2, nm.t_1q))
    tr2 = thermal_relaxation_error(nm.t1, nm.t2, nm.t_2q)
    e2 = e2.compose(tr2.expand(tr2))
    m.add_all_qubit_quantum_error(e1, ["ry", "rz", "rx"])
    m.add_all_qubit_quantum_error(e2, ["cx"])
    m.add_all_qubit_quantum_error(pauli_error([("I", 1 - nm.e0), ("X", nm.e0)]), ["reset"])
    for topology in ("ring", "all"):
        spec = CircuitSpec(n, 2, topology, "XYZ")
        th = RNG.uniform(0, 2 * math.pi, (3, 3, n))
        psi = Q.tfim_ground(n, 1.3)
        ours = Q.qae_forward(spec, 1, torch.tensor(th), Q.pure_dm(torch.tensor(psi[None], dtype=torch.complex128)), nm)
        ours = ours.reshape(16, 16).numpy()
        qc = qc_qae(spec, 1, th, psi, aer_init=True); qc.save_density_matrix()
        sim_ = AerSimulator(method="density_matrix", noise_model=m)
        from qiskit import transpile
        res = sim_.run(transpile(qc, sim_, optimization_level=0)).result()
        ref = dm_to_ours(np.asarray(res.data()["density_matrix"]), n)
        ideal = Q.qae_forward(spec, 1, torch.tensor(th), Q.pure_dm(torch.tensor(psi[None], dtype=torch.complex128))).reshape(16, 16).numpy()
        assert np.max(np.abs(ours - ref)) < 1e-6
        assert np.max(np.abs(ours - ideal)) > 1e-3  # noise really acts


def test_identity_without_reset_and_fidelity():
    n = 4; spec = CircuitSpec(n, 3, "all", "XYZ")
    th = torch.tensor(RNG.uniform(0, 6, (4, 3, n)))
    psi = torch.tensor(Q.tfim_ground(n, 0.9)[None], dtype=torch.complex128)
    rho = Q.pure_dm(psi)
    out = Q._apply_gates(Q._apply_gates(rho, Q.qae_gates(spec), th, None, None, False), Q.qae_gates(spec), th, None, None, True)
    assert float((out - rho).abs().max()) < 1e-12  # decoder is the exact inverse
    assert abs(float(Q.fidelity(psi, rho)[0]) - 1) < 1e-12


def test_input_channel_matches_formula():
    n = 4; psi = Q.tfim_ground(n, 1.1); p = 0.1
    r = Q.input_channel(Q.pure_dm(torch.tensor(psi[None], dtype=torch.complex128)), p).reshape(16, 16).numpy()
    P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    rho = np.outer(psi, psi).astype(complex)
    for q in range(n):
        acc = (1 - p) * rho
        for s in P[1:]:
            o = Q._op(s, q, n); acc = acc + p / 3 * o @ rho @ o.conj().T
        rho = acc
    assert np.max(np.abs(r - rho)) < 1e-12
    assert abs(np.trace(r).real - 1) < 1e-12


def test_reset_error_formula():
    rho = Q.pure_dm(torch.tensor(np.eye(4)[[1]], dtype=torch.complex128))  # |01>
    out = Q.reset_qubit(rho, 1, 0.02).reshape(4, 4).numpy().real
    assert abs(out[0, 0] - 0.98) < 1e-12 and abs(out[1, 1] - 0.02) < 1e-12


def test_tfim_ground_vs_known():
    # h -> 0: GHZ-like (|0000> + |1111>)/sqrt2 ; h large: |++++>
    g = Q.tfim_ground(4, 0.05); assert abs(abs(g[0]) ** 2 + abs(g[-1]) ** 2 - 1) < 0.01
    g = Q.tfim_ground(4, 50.0); assert abs(abs(g @ (np.ones(16) / 4)) - 1) < 1e-3


def test_splits_and_guard():
    s = Q.QSplits(4, mode="pilot")
    assert {k: len(v) for k, v in s.h.items()} == Q.SPLIT_N
    s.states("train"); s.states("val")
    with pytest.raises(Q.TestAccessError):
        s.states("test")
    assert "test" not in s.accessed


def test_gradients_vs_parameter_shift():
    n = 4; spec = CircuitSpec(n, 2, "ring", "YZ")
    psi = torch.tensor(Q.tfim_ground(n, 0.8)[None], dtype=torch.complex128)
    rho = Q.input_channel(Q.pure_dm(psi), 0.1)
    th = torch.tensor(RNG.uniform(0, 6, (3, 2, n)), requires_grad=True)
    f = lambda t: Q.fidelity(psi, Q.qae_forward(spec, 1, t, rho))[0]
    f(th).backward()
    for idx in [(0, 0, 0), (1, 1, 3), (0, 1, 2)]:
        tp = th.detach().clone(); tp[idx] += math.pi / 2
        tm = th.detach().clone(); tm[idx] -= math.pi / 2
        # each parameter appears twice (encoder +theta, decoder -theta): shift rule on the joint function needs
        # finite differences instead; use central difference with small eps
        e = 1e-6
        tp = th.detach().clone(); tp[idx] += e; tm = th.detach().clone(); tm[idx] -= e
        fd = (f(tp) - f(tm)) / (2 * e)
        assert abs(float(th.grad[idx]) - float(fd)) < 1e-6


def test_genome_decode_and_f2():
    dec = Q.make_decode_q(6)
    ph = dec(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert (ph.topology, ph.depth, ph.rotset, ph.k) == ("none", 6, "XYZ", 2) and abs(ph.lr - 0.1) < 1e-12
    ph = dec(np.array([0.6, 0.0, 0.0, 0.0, 0.0]))
    assert ph.topology == "all" and Q.f2_q(ph) == 2 * 15 and abs(ph.lr - 1e-3) < 1e-12
    assert Q.f2_max_q(6) == 180 and Q.f2_max_q(4) == 72


def test_problem_budget_with_q_decoder():
    from qmiqcae.problem import Problem
    from qmiqcae.experiment_q import run_search_q
    for m in ("mopso", "nsga2", "moead", "random", "pso", "hho", "bhho"):
        fe = lambda ph, i: {"f1": 0.5 + 0.01 * ph.depth - 0.02 * (ph.topology != "none") + 0.001 * Q.f2_q(ph)}
        p = Problem(fe, budget=40, decode_fn=Q.make_decode_q(4), f2_fn=Q.f2_q)
        run_search_q(m, p, seed=1, n=4)
        assert p.n_unique == 40, m


def test_training_learns_small():
    torch.manual_seed(0)
    s = Q.QSplits(4, mode="pilot")
    tr, va = s.states("train")[:16], s.states("val")[:8]
    m = Q.QAE(CircuitSpec(4, 2, "linear", "YZ"), 1, seed=0)
    f0 = float(Q.eval_fidelity(m, va, 0.1).mean())
    info = Q.train_qae(m, tr, va, Q.QTrainCfg(lr_max=0.05, epochs=15, batch=8))
    assert info["best_val_fid"] > f0 + 0.02


def test_unitary_path_equals_gate_path():
    for n, k in ((4, 1), (6, 2)):
        spec = CircuitSpec(n, 3, "ring", "XYZ")
        th = torch.tensor(RNG.uniform(0, 6, (4, 3, n)))
        psi = torch.tensor(np.stack([Q.tfim_ground(n, h) for h in (0.5, 1.5)]), dtype=torch.complex128)
        rho = Q.input_channel(Q.pure_dm(psi), 0.1)
        a = Q._qae_ideal_unitary(spec, k, th, rho)
        g = Q._apply_gates(rho, Q.qae_gates(spec), th, None, None, False)
        for q in range(k, n):
            g = Q.reset_qubit(g, q)
        g = Q._apply_gates(g, Q.qae_gates(spec), th, None, None, True)
        assert float((a - g).abs().max()) < 1e-12


def test_d20_new_ansatz_contains_old_and_disentangles_ghz():
    """Regression: (i) with the initial layer at 0 the new ansatz equals the old one, so the old circuit family is
    contained (training can only do as well or better); (ii) the old ansatz had no rotation before the first CNOT."""
    n = 6
    spec = CircuitSpec(n, 2, "linear", "YZ")
    th_old = torch.tensor(RNG.uniform(0, 6, (2, 2, n)))
    th_new = torch.cat([torch.zeros(1, 2, n, dtype=torch.float64), th_old])
    psi = torch.tensor(np.stack([Q.tfim_ground(n, h) for h in (0.4, 1.6)]), dtype=torch.complex128)
    rho = Q.input_channel(Q.pure_dm(psi), 0.2)
    new = Q.qae_forward(spec, 1, th_new, rho)
    Q.PRE_ROT = False
    try:
        old = Q.qae_forward(spec, 1, th_old, rho)
        assert Q.qae_gates(spec)[0][0] == "CNOT"
    finally:
        Q.PRE_ROT = True
    assert float((new - old).abs().max()) < 1e-12
    assert Q.qae_gates(spec)[0][0] == "ROT" and Q.qae_gates(spec)[n * 2][0] == "CNOT"
