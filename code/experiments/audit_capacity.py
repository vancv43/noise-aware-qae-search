# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Capacity audit of the zero-latent (fixed-output) model of roadmap_v2.py. Code check only, not a reported result:
can training reach the optimal constant state (leading eigenvector of the mean clean TRAINING state) when the
circuit is expressive enough? Grid: topology x depth 1-6 x lr {0.03, 0.1}, seed 0, ideal training as everywhere.
Records ideal validation/test fidelity, noisy validation fidelity, and the optimal constant reference.
Usage: python audit_capacity.py OUTDIR [--shard I N] [--dry]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import sim, qae as Q
import roadmap_v2 as R2


def optimum(n, sp):
    tr = sp.states("train"); rho = torch.einsum("bi,bj->ij", tr, tr.conj()) / len(tr)
    phi = torch.linalg.eigh(rho)[1][:, -1]
    return {s: float((torch.abs(sp.states(s).conj() @ phi) ** 2).mean()) for s in ("val", "test")}


if __name__ == "__main__":
    outdir = sys.argv[1]; dry = "--dry" in sys.argv
    si, sn = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
    torch.set_num_threads(1); sim.set_precision("c64"); os.makedirs(outdir, exist_ok=True)
    its = [(n, t, d, lr) for n in (6, 4) for t in ("ring", "linear", "all") for d in range(1, 7) for lr in (0.03, 0.1)]
    if dry:
        its = its[:2]
    OUT = os.path.join(outdir, f"capacity_shard{si}.jsonl")
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    for j, (n, t, d, lr) in enumerate(its):
        key = f"n{n}/{t}/p{d}/lr{lr}"
        if j % sn != si or key in done:
            continue
        t0 = time.time(); sp = Q.QSplits(n, mode="final")
        spec = Q.QPhenotype(n=n, topology=t, depth=d, rotset="XYZ", k=1, lr=lr).spec
        m, info = R2.train0(spec, lr, 0, 2 if dry else 100, sp)
        rec = {"key": key, "n": n, "val_ideal": info["best_val_fid"], "best_epoch": info["best_epoch"],
               "test_ideal": float(Q.eval_fidelity(m, sp.states("test"), 0.2).mean()),
               "val_noisy": float(Q.eval_fidelity(m, sp.states("val"), 0.2, noise=R2.MAIN).mean()),
               "prep_cnot": Q.n_cnot_qae(spec) // 2, "optimum": optimum(n, sp), "seconds": time.time() - t0, "dry": dry}
        open(OUT, "a").write(json.dumps(rec) + "\n"); print(key, round(time.time() - t0, 1), flush=True)
