# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Pilot (gates G1a', G1c', G1d', G1e'), validation only. Usage: python g1v2_pilot.py OUTDIR [--dry]."""
import json, os, sys, time, traceback
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae import sim, qae as Q
from qmiqcae.circuits import CircuitSpec
outdir = sys.argv[1]; DRY = "--dry" in sys.argv
os.makedirs(outdir, exist_ok=True)
torch.set_num_threads(int(os.environ.get("NTHREADS", "4"))); sim.set_precision("c64")
OUT = os.path.join(outdir, "g1v2.jsonl")
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
EPOCHS = 2 if DRY else 100
NG = 2 if DRY else 30
NOISE = sim.NoiseModel(); P_IN = 0.2


def run(key, fn):
    if key in done: return
    t0 = time.time()
    try:
        rec = fn(); rec["status"] = "ok"
    except Exception as e:
        rec = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()}
    rec.update({"key": key, "seconds": time.time() - t0, "smoke": DRY, "code_hash": os.environ.get("CODE_HASH", "unknown"),
                "precision": "c64", "stage": "pilot_v2"})
    open(OUT, "a").write(json.dumps(rec, default=float) + "\n")
    print(key, {k: rec.get(k) for k in ("val_fid_ideal", "f1", "f2", "floor", "status")}, round(rec["seconds"], 1), flush=True)


def train_eval(n, ph_like, seed, splits):
    tr, va = splits.states("train"), splits.states("val")
    m = Q.QAE(ph_like.spec, ph_like.k, seed=seed)
    info = Q.train_qae(m, tr, va, Q.QTrainCfg(lr_max=ph_like.lr, epochs=EPOCHS, p_in=P_IN, seed=seed))
    t1 = time.time(); fn_ = float(Q.eval_fidelity(m, va, P_IN, noise=NOISE).mean()); te = time.time() - t1
    floor = float(Q.fidelity(va, Q.input_channel(Q.pure_dm(va), P_IN)).mean())
    return {"n": n, "phenotype": ph_like.key(), "val_fid_ideal": info["best_val_fid"], "val_fid_noisy": fn_, "f1": 1 - fn_,
            "f2": Q.f2_q(ph_like), "floor": floor, "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
            "flags": info["flags"], "train_seconds": info["seconds"], "eval_seconds": te,
            "history": [{k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()} for h in info["history"]]}


for n in (4, 6):
    sp = Q.QSplits(n, mode="pilot")
    K = Q.QPhenotype(n, "linear", 3, "YZ", 1, 3e-2)
    for seed in (900, 901, 902):
        run(f"G1a/n{n}/{seed}", lambda: {"gate": "G1a", "seed": seed, **train_eval(n, K, seed, sp)})
    G = np.random.default_rng(960).random((30, Q.D_Q))
    dec = Q.make_decode_q(n)
    for gi in range(NG):
        ph = dec(G[gi])
        run(f"G1c/n{n}/{gi}", lambda: {"gate": "G1c", "genome_idx": gi, "seed": gi, **train_eval(n, ph, gi, sp)})
    for gi in range(min(4, NG)):
        ph = dec(G[gi])
        for rs in (1, 2):  # rs 0 = the G1c run itself (seed gi)
            run(f"G1d/n{n}/{gi}/{rs}", lambda: {"gate": "G1d", "genome_idx": gi, "seed": rs * 100003 + gi,
                                               **train_eval(n, ph, rs * 100003 + gi, sp)})
    assert "test" not in sp.accessed
print("PART DONE v2", flush=True)
