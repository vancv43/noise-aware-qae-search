# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Frozen v2 gate rules (the study design) -> results/g1v2/g1v2_verdict.json"""
import glob, json, os
import numpy as np
from scipy import stats
R = os.path.join(os.path.dirname(__file__), "..", "..", "results", os.environ.get("G1DIR", "g1v2"))
recs = [json.loads(l) for f in glob.glob(os.path.join(R, "**", "g1v2.jsonl"), recursive=True) for l in open(f)]
recs = [r for r in recs if not r.get("smoke")]
V = {"n_records": len(recs), "failed": [r["key"] for r in recs if r["status"] != "ok"]}
ok = [r for r in recs if r["status"] == "ok"]
a = [r for r in ok if r["gate"] == "G1a"]
V["G1a"] = {r["key"]: [round(r["val_fid_ideal"], 4), round(r["floor"] + 0.05, 4), r["flags"]] for r in a}
V["G1a_pass"] = len(a) == 6 and all(r["val_fid_ideal"] >= r["floor"] + 0.05 and not r["flags"]["underfit"] for r in a)
V["G1c"] = {}
for n in (4, 6):
    c = [r for r in ok if r["gate"] == "G1c" and r["n"] == n]
    F = np.array([[r["f1"], r["f2"]] for r in c])
    nd = [i for i in range(len(F)) if not any((F[j] <= F[i]).all() and (F[j] < F[i]).any() for j in range(len(F)))]
    rho = float(stats.spearmanr(F[:, 0], F[:, 1])[0])
    V["G1c"][f"n{n}"] = {"n_genomes": len(c), "front": sorted({(round(F[i, 0], 4), int(F[i, 1])) for i in nd}, key=lambda x: x[1]),
                         "n_nondominated": len(nd), "spearman": round(rho, 3), "pass": len(nd) >= 3 and rho < 0.8,
                         "best_noisy_fid": round(1 - F[:, 0].min(), 4)}
    d = {}
    for r in ok:
        if r["n"] == n and (r["gate"] == "G1d" or (r["gate"] == "G1c" and r["genome_idx"] < 4)):
            d.setdefault(r["genome_idx"], []).append(r["f1"])
    w = float(np.mean([np.std(v, ddof=1) for v in d.values()])); b = float(np.std([np.mean(v) for v in d.values()], ddof=1))
    V.setdefault("G1d", {})[f"n{n}"] = {"within_sd": round(w, 4), "between_sd": round(b, 4), "pass": w < 0.5 * b}
    sec = [r["train_seconds"] + r["eval_seconds"] for r in c]
    V.setdefault("G1e", {})[f"n{n}"] = {"mean_s": round(float(np.mean(sec)), 1), "max_s": round(float(np.max(sec)), 1)}
V["G1c_pass"] = all(v["pass"] for v in V["G1c"].values())
V["G1d_pass"] = all(v["pass"] for v in V["G1d"].values())
V["G1e_full_search_cpu_h"] = round(sum(8 * 10 * 100 * v["mean_s"] for v in V["G1e"].values()) / 3600, 1)
json.dump(V, open(os.path.join(R, "g1v2_verdict.json"), "w"), indent=1)
print(json.dumps(V, indent=1))
