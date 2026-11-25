# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Final evaluation (the study design) of the selected point of every search run. The ONLY script that opens the test split
(mode 'final'). It rebuilds the exact searched model (same seed = run_seed*100003 + eval, same training), asserts that the
reproduced ideal val fidelity equals the search record, then scores test fidelity under each condition.
Usage: python final_eval_v2.py SELECTED_JSON(analysis.json) OUTDIR [--dry] [--shard I N]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae import sim, qae as Q
MAIN = sim.NoiseModel()
REPRO_TOL = 1e-5


def conditions():
    c = {"ideal": (None, 0.2), "main": (MAIN, 0.2)}
    for p2 in (0.005, 0.01, 0.02, 0.05):
        c[f"p2={p2:g}"] = (sim.NoiseModel(p1=p2 / 10, p2=p2), 0.2)
    for t in (50e-6, 100e-6, 200e-6):
        c[f"T1=T2={t * 1e6:g}us"] = (sim.NoiseModel(t1=t, t2=t), 0.2)
    for e in (0.0, 0.02, 0.05):
        c[f"reset={e:g}"] = (sim.NoiseModel(e0=e, e1=e), 0.2)
    for pin in (0.1, 0.2, 0.3):
        c[f"p_in={pin:g}"] = (MAIN, pin)
    return c


def evaluate_selected(sel_item, n, epochs=100, dry=False):
    # identical numerics to search_v2.py (without this the rebuilt model differed by 8e-8)
    sim.set_precision("c64"); torch.set_num_threads(1)
    ph = Q.QPhenotype(**sel_item["phenotype"])
    seed = sel_item["run_seed"] * 100003 + sel_item["eval"]
    sp = Q.QSplits(n, mode="final")
    tr, va = sp.states("train"), sp.states("val")
    m = Q.QAE(ph.spec, ph.k, seed=seed)
    info = Q.train_qae(m, tr, va, Q.QTrainCfg(lr_max=ph.lr, epochs=epochs, patience=epochs + 1, p_in=0.2, seed=seed))
    repro = abs(info["best_val_fid"] - sel_item["val_fid_ideal"])
    # tolerance 1e-5 = ~100 ulp, 400x below the seed SD 0.004. A model beyond it is NOT scored on test.
    if not dry and repro > REPRO_TOL:
        return {"repro_abs_diff": repro, "status": "not_reproduced", "test": {}, "floor": {}}
    te = sp.states("test")
    out = {"repro_abs_diff": repro, "status": "ok", "test": {}, "floor": {}}
    for name, (nm, pin) in conditions().items():
        out["test"][name] = float(Q.eval_fidelity(m, te, pin, noise=nm).mean())
    for pin in (0.1, 0.2, 0.3):
        out["floor"][f"p_in={pin:g}"] = float(Q.fidelity(te, Q.input_channel(Q.pure_dm(te), pin)).mean())
    return out


if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]; dry = "--dry" in sys.argv
    torch.set_num_threads(1); sim.set_precision("c64"); os.makedirs(outdir, exist_ok=True)
    si, sn = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
    A = json.load(open(src)); OUT = os.path.join(outdir, f"final_eval_shard{si}.jsonl")
    items = [(n, key, s) for n, r in sorted(A["per_n"].items()) for key, s in sorted(r["selected"].items())]
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    for j, (n, key, s) in enumerate(items):
        k = f"n{n}/{key}"
        if j % sn != si or k in done:
            continue
        t0 = time.time(); res = evaluate_selected(s, int(n), epochs=2 if dry else 100, dry=dry)
        open(OUT, "a").write(json.dumps({"key": k, "n": int(n), **s, **res, "seconds": time.time() - t0, "smoke": dry}) + "\n")
        print(k, res["status"], res["repro_abs_diff"], flush=True)  # no test numbers in the log
