# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""C1/C2 search runs (the study design). Usage: python search_v2.py N ARM SEEDS(comma) OUTDIR [--dry]
One JSONL of evaluations per (n, arm, seed) + one summary line in runs.jsonl. A run whose summary exists is skipped
(resume). Reads train/val only (mode 'search')."""
import json, os, sys, time, traceback
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae import sim, qae as Q
from qmiqcae.problem import Problem
from qmiqcae.experiment_q import make_search_evaluator_q, run_search_q, MAIN_NOISE
n, arm, seeds, outdir = int(sys.argv[1]), sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
DRY = "--dry" in sys.argv
torch.set_num_threads(int(os.environ.get("NTHREADS", "1"))); sim.set_precision("c64")
os.makedirs(outdir, exist_ok=True)
SUM = os.path.join(outdir, "runs.jsonl")
done = {(r["n"], r["arm"], r["seed"]) for r in map(json.loads, open(SUM))} if os.path.exists(SUM) else set()
method = "mopso" if arm in ("namopso", "mopso_ideal") else arm
noise = None if arm == "mopso_ideal" else MAIN_NOISE
BUDGET = 6 if DRY else 100
EPOCHS = 2 if DRY else 100
for seed in seeds:
    if (n, arm, seed) in done:
        continue
    log = os.path.join(outdir, f"evals_n{n}_{arm}_s{seed}.jsonl")
    if os.path.exists(log):
        os.remove(log)  # incomplete run from a killed session: restart it (no partial reuse)
    sp = Q.QSplits(n, mode="search")
    fe = make_search_evaluator_q(sp, noise, EPOCHS, run_seed=seed)
    prob = Problem(fe, budget=BUDGET, log_path=log, decode_fn=Q.make_decode_q(n), f2_fn=Q.f2_q)
    t0 = time.time()
    try:
        out = run_search_q(method, prob, seed=seed, n=n); status = "ok"
    except Exception as e:
        out = {"error": repr(e), "trace": traceback.format_exc()}; status = "failed"
    assert "test" not in sp.accessed
    rec = {"n": n, "arm": arm, "method": method, "seed": seed, "status": status, "seconds": time.time() - t0,
           "problem": prob.summary(), "smoke": DRY, "code_hash": os.environ.get("CODE_HASH", "unknown"),
           "noise": MAIN_NOISE.tag(), "p_in": 0.2, "epochs": EPOCHS, "precision": "c64",
           "algo_out": out if status != "ok" else {k: v for k, v in out.items() if k != "problem"}}
    open(SUM, "a").write(json.dumps(rec, default=float) + "\n")
    print("RUN DONE", n, arm, seed, status, prob.summary(), round(rec["seconds"]), flush=True)
