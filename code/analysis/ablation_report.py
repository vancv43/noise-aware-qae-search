# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Merge the post hoc selection-rule test scores (results/ablation/out) with the frozen final evaluation
(results/final_v2/out) and print per arm x rule means, plus paired Wilcoxon-Holm tests and A12 (paired by seed)."""
import glob, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae.analysis import a12, holm
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
fin = {json.loads(l)["key"]: json.loads(l) for f in glob.glob(f"{R}/final_v2/out/*.jsonl") for l in open(f)}
abl = {json.loads(l)["key"]: json.loads(l) for f in glob.glob(f"{R}/ablation/out/*.jsonl") for l in open(f)}
val = json.load(open(f"{R}/ablation/selection_val.json"))
bad = [k for k, v in abl.items() if v["status"] != "ok"]
print("ablation models", len(abl), "not ok", bad, "max repro", max(v["repro_abs_diff"] for v in abl.values()))
by_eval = {}
for k, v in abl.items():
    n, arm_s = k.split("/", 1); arm, s = arm_s.split("@")[0].split("/")
    by_eval[(n, arm, s, v["eval"])] = v
def rec(n, arm, s, rule):
    it = val[f"n{n}/{arm}/{s}/{rule}"]
    if it["same_as_cheap"]:
        return fin[f"n{n}/{arm}/{s}"]
    return by_eval[(f"n{n}", arm, s, it["eval"])]
out = {}
for n in ("4", "6"):
    for arm in ("namopso", "mopso_ideal", "nsga2", "moead", "random"):
        for rule in ("cheap", "maxfid", "ideal"):
            rs = [rec(n, arm, f"s{s}", rule) for s in range(10)]
            v = [val[f"n{n}/{arm}/s{s}/{rule}"] for s in range(10)]
            d = {"f2": np.mean([x["f2"] for x in v]), "val": np.mean([x["val_fid_noisy"] for x in v]),
                 "main": [x["test"]["main"] for x in rs], "p05": [x["test"]["p2=0.05"] for x in rs],
                 "beat": sum(x["test"]["main"] > x["floor"]["p_in=0.2"] for x in rs), "floor": rs[0]["floor"]["p_in=0.2"]}
            out[(n, arm, rule)] = d
            print(f"n{n} {arm:12s} {rule:7s} f2 {d['f2']:5.1f} val {d['val']:.3f} main {np.mean(d['main']):.3f}±{np.std(d['main'],ddof=1):.3f}"
                  f" p05 {np.mean(d['p05']):.3f}±{np.std(d['p05'],ddof=1):.3f} beat {d['beat']}/10 floor {d['floor']:.3f}")
    # tests: NA-QAS-QAE with each rule vs MOPSO-ideal with each rule, plus NA rule vs NA cheap
    for cond in ("main", "p05"):
        P, E = {}, {}
        x = out[(n, "namopso", "maxfid")][cond]
        for lab, (arm, rule) in {"NA-cheap": ("namopso", "cheap"), "ideal-cheap": ("mopso_ideal", "cheap"),
                                 "ideal-maxfid": ("mopso_ideal", "maxfid"), "ideal-ideal": ("mopso_ideal", "ideal")}.items():
            y = out[(n, arm, rule)][cond]
            P[lab] = stats.wilcoxon(x, y).pvalue if np.any(np.array(x) != np.array(y)) else 1.0
            E[lab] = a12(x, y)
        H = holm(P)
        print(f"  n{n} {cond} NA-maxfid vs:", {k: (round(H[k], 4), round(E[k], 2)) for k in P})
# per method x rule summary (selection-rule table of the Appendix), saved for reproduction
json.dump({f"n{n}/{arm}/{rule}": {"f2": round(float(d["f2"]), 1), "val": round(float(d["val"]), 3),
                                  "test_main": round(float(np.mean(d["main"])), 3), "test_p2_0.05": round(float(np.mean(d["p05"])), 3),
                                  "beat_floor": int(d["beat"])} for (n, arm, rule), d in out.items()},
          open(f"{R}/ablation/rule_summary.json", "w"), indent=1)
