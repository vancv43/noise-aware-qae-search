# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Data-only revision analyses (no training, no new test reading beyond stored results)."""
import glob, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import qae as Q
from qmiqcae.analysis import a12, holm, hypervolume
from analyze_search_v2 import load, objectives
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
runs, evals = load(os.path.join(R, "search_v2"))
A = json.load(open(f"{R}/search_v2/analysis.json"))
fin = {json.loads(l)["key"]: json.loads(l) for f in glob.glob(f"{R}/final_v2/out/*.jsonl") for l in open(f)}
rng = np.random.default_rng(0)
def a12_ci(x, y, B=2000):
    x, y = np.asarray(x), np.asarray(y); bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x)); bs.append(a12(x[i], y[i]))   # paired by seed
    return a12(x, y), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
out = {}
# 1 A12 with paired bootstrap CI for HV and test fidelity, NA vs MOPSO-ideal and PSO
for n in ("4", "6"):
    hv = A["per_n"][n]["hv"]
    for arm in ("mopso_ideal", "pso", "nsga2"):
        r = a12_ci(hv["namopso"]["values"], hv[arm]["values"]); out[f"hv n{n} vs {arm}"] = [round(v, 2) for v in r]
    for cond in ("main", "p2=0.05"):
        x = [fin[f"n{n}/namopso/s{s}"]["test"][cond] for s in range(10)]
        for arm in ("mopso_ideal", "pso"):
            y = [fin[f"n{n}/{arm}/s{s}"]["test"][cond] for s in range(10)]
            out[f"test {cond} n{n} vs {arm}"] = [round(v, 2) for v in a12_ci(x, y)]
# 2 PSO short runs excluded
short = {"4": 9, "6": 6}
for n in ("4", "6"):
    hv = A["per_n"][n]["hv"]; keep = [s for s in range(10) if s != short[n]]
    x = np.array(hv["namopso"]["values"])[keep]; y = np.array(hv["pso"]["values"])[keep]
    out[f"pso excl short n{n}"] = {"p_raw": round(float(stats.wilcoxon(x, y).pvalue), 3), "a12": round(a12(x, y), 2),
                                   "pso_median": round(float(np.median(y)), 4), "na_median": round(float(np.median(x)), 4)}
# 3 f_SO weight: post hoc re-selection from the evaluated sets of the single-objective methods
for n in (4, 6):
    fm = Q.f2_max_q(n)
    for w in (0.05, 0.1, 0.2):
        cn, vf = [], []
        for arm in ("pso", "hho", "bhho"):
            for s in range(10):
                ev = [e for e in evals[(n, arm, s)] if e["status"] == "ok"]
                b = min(ev, key=lambda e: (1 - e["val_fid_noisy"]) + w * e["f2"] / fm)
                cn.append(b["f2"]); vf.append(b["val_fid_noisy"])
        out[f"fso w={w} n{n}"] = {"mean_cnot": round(float(np.mean(cn)), 1), "mean_noisy_val": round(float(np.mean(vf)), 3)}
# 4 learning rate vs late peak (best epoch >= 96) over all evaluations
for n in (4, 6):
    allr = [e for k in runs if k[0] == n for e in evals[k] if e["status"] == "ok"]
    lr = np.array([e["phenotype"]["lr"] for e in allr]); late = np.array([e["best_epoch"] >= 96 for e in allr])
    q = np.quantile(lr, [0, 1/3, 2/3, 1])
    out[f"late-peak share by lr tercile n{n}"] = [round(float(late[(lr >= q[i]) & (lr <= q[i+1])].mean()), 2) for i in range(3)]
    out[f"lr terciles n{n}"] = [round(float(v), 4) for v in q]
# 5 all-to-all share in selected models
for n in ("4", "6"):
    for arm in ("namopso", "mopso_ideal", "nsga2", "moead", "random", "pso", "hho", "bhho"):
        tops = [fin[f"n{n}/{arm}/s{s}"]["phenotype"]["topology"] for s in range(10)]
        out.setdefault(f"all-to-all selected n{n}", {})[arm] = tops.count("all")
# 6 coverage of the 96 structures per run
for n in (4, 6):
    for arm in ("namopso", "random", "nsga2"):
        cov = [len({(e["phenotype"]["topology"], e["phenotype"]["depth"] if e["phenotype"]["topology"] != "none" else 0,
                     e["phenotype"]["rotset"], e["phenotype"]["k"]) for e in evals[(n, arm, s)]}) for s in range(10)]
        out[f"distinct structures per run n{n} {arm}"] = round(float(np.mean(cov)), 1)
json.dump(out, open(f"{R}/revision/dataonly.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
