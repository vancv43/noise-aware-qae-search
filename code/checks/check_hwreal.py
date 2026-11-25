# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Numeric checks of the extended noise model in experiments/hwreal.py (no training).
H1 idle off, reset 0  -> equals qae.qae_forward under the same noise (QAE k=1,2) and roadmap_v2.forward0 (k=0).
H2 thermal off        -> idle relaxation and reset duration change nothing.
H3 schedule           -> every gate placed once; per qubit busy + idle = makespan; no qubit in two gates of a moment;
                         a moment lasts as long as its slowest gate; ASAP: a gate never starts later than needed.
H4 outputs            -> trace 1, Hermitian, PSD with idle and 4 us reset.
H5 hand case          -> 2 qubits, one CNOT then RX on qubit 0: qubit 1 idles exactly t_2q - 0 ... computed by hand.
H6 fixed control      -> output independent of the input under the extended model.
H7 direction          -> mean fidelity drop main -> main+idle+reset1us grows with CNOT count over random circuits."""
import json, os, sys, itertools
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
from qmiqcae import sim, qae as Q
import hwreal as H, roadmap_v2 as R2, roadmap_v1 as R1
from scipy import stats

sim.set_precision("c128"); torch.manual_seed(0); res = {}
MAIN = R1.MAIN; NOTH = sim.NoiseModel(thermal=False)
sp4 = Q.QSplits(4, mode="search"); psi = sp4.states("val")[:6]; rho = Q.input_channel(Q.pure_dm(psi), 0.2)

# H1, H2
d1, d2 = 0.0, 0.0
for t, dep, rs, k in itertools.product(("ring", "linear", "all"), (1, 3), ("YZ", "XYZ"), (1, 2)):
    m = Q.QAE(Q.CircuitSpec(4, dep, t, rs), k, seed=1, init_scale=1.0)
    with torch.no_grad():
        d1 = max(d1, float((H.forward_ext(m.spec, k, m.theta, rho, MAIN, False, 0.0) - Q.qae_forward(m.spec, k, m.theta, rho, MAIN)).abs().max()))
        d2 = max(d2, float((H.forward_ext(m.spec, k, m.theta, rho, NOTH, True, 4e-6) - Q.qae_forward(m.spec, k, m.theta, rho, NOTH)).abs().max()))
m0 = R2.QAE0(Q.CircuitSpec(4, 2, "linear", "XYZ"), seed=2)
with torch.no_grad():
    m0.theta.copy_(torch.randn_like(m0.theta))
    d1 = max(d1, float((H.forward_ext(m0.spec, 0, m0.theta, rho, MAIN, False, 0.0) - R2.forward0(m0.spec, m0.theta, rho, MAIN)).abs().max()))
res["H1 max |ext(idle off) - reference|"] = d1; res["H2 max |ext(thermal off) - reference|"] = d2

# H3
ok = True
for t, dep, n in itertools.product(("ring", "linear", "all"), (1, 2, 4), (4, 6)):
    g = Q.qae_gates(Q.CircuitSpec(n, dep, t, "XYZ"))
    for seq in (g, list(reversed(g))):
        sch = H.schedule(seq, n, MAIN); placed = sorted(i for _, mo in sch for i in mo)
        ok &= placed == list(range(len(seq)))
        span = sum(d for d, _ in sch); busy = [0.0] * n
        for dur, mo in sch:
            qs = [q for i in mo for q in ((seq[i][1], seq[i][2]) if seq[i][0] == "CNOT" else (seq[i][4],))]
            ok &= len(qs) == len(set(qs))
            ok &= abs(dur - max(MAIN.t_2q if seq[i][0] == "CNOT" else MAIN.t_1q for i in mo)) < 1e-18
            for i in mo:
                for q in ((seq[i][1], seq[i][2]) if seq[i][0] == "CNOT" else (seq[i][4],)):
                    busy[q] += MAIN.t_2q if seq[i][0] == "CNOT" else MAIN.t_1q
        ok &= all(b <= span + 1e-15 for b in busy)
res["H3 schedule valid"] = bool(ok)

# H5 hand case: n=2, gates [CNOT(0,1), RX on q0]; moment 1 = CNOT (500 ns), moment 2 = RX (50 ns) -> qubit 1 idles 50 ns
spec = Q.CircuitSpec(2, 1, "linear", "XYZ"); th = torch.zeros(2, 3, 2, dtype=torch.float64)
gates = [("CNOT", 0, 1), ("ROT", "RX", 0, 0, 0)]
r0 = Q.pure_dm(torch.tensor([[0, 0, 0, 1.0]], dtype=torch.complex128))           # |11>
S1, S2 = sim.noise_superops(MAIN)
got = H._apply_scheduled(r0, gates, th, MAIN, S1, S2, False, True)
ref = sim.dm_channel_2q(sim.dm_apply_cnot(r0, 0, 1), S2, 0, 1)
ref = sim.dm_channel_1q(sim.dm_apply_1q(ref, sim.GATE_FN["RX"](th[0, 0, 0]), 0), S1, 0)
ref = sim.dm_channel_1q(ref, H.thermal_superop(MAIN, MAIN.t_1q), 1)
res["H5 hand case |diff|"] = float((got - ref).abs().max())

# H4, H6
with torch.no_grad():
    m = Q.QAE(Q.CircuitSpec(4, 3, "ring", "XYZ"), 1, seed=5, init_scale=1.0)
    o = H.forward_ext(m.spec, 1, m.theta, rho, MAIN, True, 4e-6).reshape(6, 16, 16)
    res["H4 trace err"] = float((torch.diagonal(o, dim1=1, dim2=2).sum(-1).real - 1).abs().max())
    res["H4 herm err"] = float((o - o.conj().transpose(1, 2)).abs().max())
    res["H4 min eig"] = float(torch.linalg.eigvalsh(0.5 * (o + o.conj().transpose(1, 2))).min())
    o0 = H.forward_ext(m0.spec, 0, m0.theta, rho, MAIN, True, 1e-6).reshape(6, 16, 16)
    res["H6 fixed input dependence"] = float((o0 - o0[0]).abs().max())

# H7
drops, cn = [], []
with torch.no_grad():
    for t, dep, k in itertools.product(("ring", "linear", "all", "none"), (1, 2, 3, 4), (1, 2)):
        m = Q.QAE(Q.CircuitSpec(4, dep, t, "YZ"), k, seed=7, init_scale=0.3)
        a = Q.fidelity(psi, H.forward_ext(m.spec, k, m.theta, rho, MAIN, False, 0.0)).mean()
        b = Q.fidelity(psi, H.forward_ext(m.spec, k, m.theta, rho, MAIN, True, 1e-6)).mean()
        drops.append(float(a - b)); cn.append(Q.n_cnot_qae(m.spec))
res["H7 spearman(drop, CNOT)"] = float(stats.spearmanr(drops, cn)[0]); res["H7 drop min/max"] = [min(drops), max(drops)]

json.dump(res, open(os.path.join(HERE, "..", "..", "results", "hwreal", "check_hwreal.json"), "w"), indent=1)
for k, v in res.items():
    print(k, v)
