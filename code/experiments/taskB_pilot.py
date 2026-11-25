# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Option B pilot (pre-specified in results/taskB_pilot/PRESPEC.md). Validation states only.
Families F3 (mixed-field Ising) and F5 (TFIM/XXZ mixture); QAE grid and fixed-output control under the main noise.
Usage: python taskB_pilot.py OUTDIR [--shard I N] [--dry]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import sim, qae as Q
import roadmap_v1 as R1, roadmap_v2 as R2

X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1., -1.]).astype(complex); Y = np.array([[0, -1j], [1j, 0]]); I2 = np.eye(2)
SPLIT = {"train": (64, 21), "val": (32, 22)}          # new seeds; the test split is not generated here


def op(o, q, n):
    m = np.array([[1.0 + 0j]])
    for j in range(n):
        m = np.kron(m, o if j == q else I2)
    return m


def ham(fam, n, a):
    zz = sum(op(Z, q, n) @ op(Z, (q + 1) % n, n) for q in range(n))
    if fam == "F3":
        h, g = a
        return -zz - h * sum(op(X, q, n) for q in range(n)) - g * sum(op(Z, q, n) for q in range(n))
    kind, h, D = a
    if kind == 0:
        return -zz - h * sum(op(X, q, n) for q in range(n))
    return sum(op(X, q, n) @ op(X, (q + 1) % n, n) + op(Y, q, n) @ op(Y, (q + 1) % n, n) for q in range(n)) + D * zz


def params(fam, rng):
    if fam == "F3":
        return (rng.uniform(0.2, 1.2), rng.uniform(-0.3, 0.3))
    return (int(rng.integers(2)), rng.uniform(0.2, 2.0), rng.uniform(-0.5, 2.0))


def states(fam, n, split):
    N, seed = SPLIT[split]; rng = np.random.default_rng(seed + (0 if fam == "F3" else 100)); out = []
    while len(out) < N:
        w, v = np.linalg.eigh(ham(fam, n, params(fam, rng)))
        if w[1] - w[0] > 1e-6:                       # skip degenerate ground levels
            s = v[:, 0]; s = s * np.exp(-1j * np.angle(s[np.argmax(np.abs(s))]))   # fix global phase
            out.append(s)
    return torch.tensor(np.stack(out), dtype=torch.complex128)


def items():
    its = []
    for fam in ("F3", "F5"):
        for n in (4, 6):
            for t in ("ring", "linear"):
                for d in (1, 2, 3):
                    for r in ("YZ", "XYZ"):
                        for k in (1, 2):
                            for lr in (0.03, 0.1):
                                its.append({"key": f"{fam}/n{n}/qae/{t}/p{d}/{r}/k{k}/lr{lr}", "fam": fam, "n": n, "kind": "qae", "t": t, "d": d, "r": r, "k": k, "lr": lr})
            for t in ("ring", "linear", "all"):
                for d in (1, 2, 3):
                    for lr in (0.03, 0.1):
                        its.append({"key": f"{fam}/n{n}/fixed/{t}/p{d}/lr{lr}", "fam": fam, "n": n, "kind": "fixed", "t": t, "d": d, "r": "XYZ", "k": 1, "lr": lr})
    return its


def run(it, epochs):
    n = it["n"]; tr, va = states(it["fam"], n, "train"), states(it["fam"], n, "val")
    ph = Q.QPhenotype(n=n, topology=it["t"], depth=it["d"], rotset=it["r"], k=it["k"], lr=it["lr"])
    cfg = Q.QTrainCfg(lr_max=it["lr"], epochs=epochs, patience=epochs + 1, p_in=0.2, seed=0)
    m = Q.QAE(ph.spec, ph.k, seed=0) if it["kind"] == "qae" else R2.QAE0(ph.spec, seed=0)
    info = Q.train_qae(m, tr, va, cfg)
    rho = torch.einsum("bi,bj->ij", tr, tr.conj()) / len(tr); phi = torch.linalg.eigh(rho)[1][:, -1]
    return {"val_ideal": info["best_val_fid"], "val_noisy": float(Q.eval_fidelity(m, va, 0.2, noise=R1.MAIN).mean()),
            "floor": R1.floor(va), "const_ideal": float((torch.abs(va.conj() @ phi) ** 2).mean()),
            "f2": Q.f2_q(ph) if it["kind"] == "qae" else Q.n_cnot_qae(ph.spec) // 2, "best_epoch": info["best_epoch"]}


if __name__ == "__main__":
    outdir = sys.argv[1]; dry = "--dry" in sys.argv
    si, sn = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
    torch.set_num_threads(1); sim.set_precision("c64"); os.makedirs(outdir, exist_ok=True)
    its = items()
    if dry:
        its = [its[0], its[-1]]
    OUT = os.path.join(outdir, f"pilot_shard{si}.jsonl")
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    for j, it in enumerate(its):
        if j % sn != si or it["key"] in done:
            continue
        t0 = time.time(); res = run(it, 2 if dry else 100)
        open(OUT, "a").write(json.dumps({**it, **res, "seconds": time.time() - t0, "dry": dry}) + "\n")
        print(it["key"], round(time.time() - t0, 1), flush=True)
