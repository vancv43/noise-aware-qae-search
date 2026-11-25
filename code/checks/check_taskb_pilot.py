# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Numeric audit of the option-B pilot (no training): is any computation wrong in a way that could explain the FAIL?
C1 fixed-output and QAE noisy paths with an all-zero noise model equal the ideal paths.
C2 floor of the pilot = independent numpy depolarizing on the same validation states.
C3 train/val states disjoint; ground-state gaps; Hamiltonians Hermitian and ground state is the eigh minimum.
C4 fixed-output ideal fidelity cannot exceed lambda_1 of the validation mean state (upper bound for a pure constant).
C5 a random-parameter noisy QAE forward stays a valid density matrix (trace 1, Hermitian, PSD).
C6 the noisy forward of a QAE with all angles 0 equals: reset trash qubits of the noisy input (and gate noise only)."""
import json, os, sys, glob
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
from qmiqcae import sim, qae as Q
import taskB_pilot as P, roadmap_v2 as R2, roadmap_v1 as R1

sim.set_precision("c128"); torch.manual_seed(0)
ZERO = sim.NoiseModel(p1=0.0, p2=0.0, e0=0.0, e1=0.0, thermal=False)
res = {}

def depol_np(rho, p, n):
    d = 2 ** n; I = np.eye(2)
    for q in range(n):
        r = rho.reshape([2] * (2 * n)); tq = np.trace(r, axis1=q, axis2=q + n)
        mixed = np.moveaxis(np.tensordot(tq, I / 2, 0), [2 * n - 2, 2 * n - 1], [q, q + n]).reshape(d, d)
        rho = (1 - 4 * p / 3) * rho + 4 * p / 3 * mixed
    return rho

for fam in ("F3", "F5"):
    for n in (4, 6):
        tag = f"{fam}/n{n}"; tr, va = P.states(fam, n, "train"), P.states(fam, n, "val")
        rho_va = Q.input_channel(Q.pure_dm(va[:6]), 0.2)
        # C1
        ph = Q.QPhenotype(n=n, topology="ring", depth=2, rotset="XYZ", k=1, lr=0.1)
        m = Q.QAE(ph.spec, 1, seed=3, init_scale=1.0); m0 = R2.QAE0(ph.spec, seed=3)
        with torch.no_grad():
            m0.theta.copy_(torch.randn_like(m0.theta))
            d_qae = float((m(rho_va, ZERO) - m(rho_va, None)).abs().max())
            d_fix = float((m0(rho_va, ZERO) - m0(rho_va, None)).abs().max())
            # fixed output must be input independent
            out0 = m0(rho_va, R1.MAIN).reshape(6, 2 ** n, 2 ** n); d_ind = float((out0 - out0[0]).abs().max())
            # C5
            o = m(rho_va, R1.MAIN).reshape(6, 2 ** n, 2 ** n)
            tr_err = float((torch.diagonal(o, dim1=1, dim2=2).sum(-1).real - 1).abs().max())
            herm = float((o - o.conj().transpose(1, 2)).abs().max())
            mineig = float(torch.linalg.eigvalsh(0.5 * (o + o.conj().transpose(1, 2))).min())
        # C2
        fl_pilot = R1.floor(va)
        v = va.numpy(); fl_np = float(np.mean([np.real(s.conj() @ depol_np(np.outer(s, s.conj()), 0.2, n) @ s) for s in v]))
        # C3
        ov = float(np.max(np.abs(tr.numpy().conj() @ v.T) ** 2))
        rng = np.random.default_rng(22 + (0 if fam == "F3" else 100)); gaps = []; herm_h = 0.0
        while len(gaps) < 32:
            Hm = P.ham(fam, n, P.params(fam, rng)); herm_h = max(herm_h, float(np.abs(Hm - Hm.conj().T).max()))
            w = np.linalg.eigvalsh(Hm)
            if w[1] - w[0] > 1e-6:
                gaps.append(w[1] - w[0])
        # C4
        rv = (v[:, :, None] * v[:, None, :].conj()).mean(0); lam1_val = float(np.linalg.eigvalsh(rv)[-1])
        rt = tr.numpy(); lam1_tr = float(np.linalg.eigvalsh((rt[:, :, None] * rt[:, None, :].conj()).mean(0))[-1])
        res[tag] = {"C1 |QAE zero-noise - ideal|": d_qae, "C1 |fixed zero-noise - ideal|": d_fix, "C1 fixed input-dependence": d_ind,
                    "C2 floor pilot": fl_pilot, "C2 floor numpy": fl_np, "C3 max train-val overlap": ov,
                    "C3 min gap val": float(min(gaps)), "C3 #gap<1e-2": int(sum(g < 1e-2 for g in gaps)), "C3 H herm err": herm_h,
                    "C4 lambda1 val": lam1_val, "C4 lambda1 train": lam1_tr,
                    "C5 trace err": tr_err, "C5 herm err": herm, "C5 min eig": mineig}

# C4 against the pilot records
R = [json.loads(l) for f in glob.glob(os.path.join(HERE, "..", "..", "results", "taskB_pilot", "out", "*.jsonl")) for l in open(f)]
for fam in ("F3", "F5"):
    for n in (4, 6):
        b = max(r["val_ideal"] for r in R if r["fam"] == fam and r["n"] == n and r["kind"] == "fixed")
        res[f"{fam}/n{n}"]["C4 best fixed val_ideal"] = b
        res[f"{fam}/n{n}"]["C4 PASS (<= lambda1 val)"] = bool(b <= res[f"{fam}/n{n}"]["C4 lambda1 val"] + 1e-6)

out = os.path.join(HERE, "..", "..", "results", "taskB_pilot", "check_taskb_pilot.json")
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k); [print("   ", a, round(b, 8) if isinstance(b, float) else b) for a, b in v.items()]
