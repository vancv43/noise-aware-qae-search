# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Analysis of the hardware-realism re-scoring exactly as in results/hwreal/PRESPEC.md (written first)."""
import json, glob, os
import numpy as np
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, "results/hwreal/out/*.jsonl"))) for l in open(f)]
assert not any(r["dry"] for r in R)
PRIM, BASE = "fresh|main+idle+reset1us", "fresh|main"
CONDS = ["fresh|main", "fresh|main+idle", "fresh|main+idle+reset1us", "fresh|main+idle+reset4us", "fresh|sher+idle+reset1us",
         "test|main", "test|main+idle", "test|main+idle+reset1us"]
out = {"records": len(R), "not_reproduced": [r["key"] for r in R if r["status"] != "ok"],
       "max_repro": max(r.get("repro_abs_diff") or 0 for r in R)}
ok = [r for r in R if r["status"] == "ok"]
mean = lambda r, c: float(np.mean(r["scores"][c]))


def group(n, arm, rule):
    """run seed -> record for a swarm group; for the control, seed -> record."""
    if arm == "fixed":
        return {r["seed"]: r for r in ok if r["n"] == n and r["arm"] == "fixed"}
    return {r["run_seed"]: r for r in ok if r["n"] == n and r["arm"] == arm and rule in r["rules"]}


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
    return adj.tolist()


G = [("NA cheap", "namopso", "cheap"), ("NA ideal", "namopso", "ideal"), ("NA maxfid", "namopso", "maxfid"),
     ("MI cheap", "mopso_ideal", "cheap"), ("MI ideal", "mopso_ideal", "ideal"), ("Control", "fixed", "-")]
for n in (4, 6):
    o = {"floor": {s: float(np.mean([r["floor"][s] for r in ok if r["n"] == n])) for s in ("test", "fresh")}}
    o["means"] = {lab: {c: round(float(np.mean([mean(r, c) for r in group(n, a, ru).values()])), 4) for c in CONDS}
                  for lab, a, ru in G}
    na = group(n, "namopso", "maxfid"); prim = {}; ps = []
    for rule in ("cheap", "ideal"):
        mi = group(n, "mopso_ideal", rule); runs = sorted(set(na) & set(mi))
        dD = [(mean(na[s], PRIM) - mean(mi[s], PRIM)) - (mean(na[s], BASE) - mean(mi[s], BASE)) for s in runs]
        p = float(stats.wilcoxon(dD).pvalue) if any(abs(x) > 0 for x in dD) else 1.0
        prim[f"vs MI {rule}"] = {"runs": len(runs), "mean_dD": round(float(np.mean(dD)), 5),
                                 "min_dD": round(min(dD), 5), "max_dD": round(max(dD), 5), "p_raw": round(p, 4)}; ps.append(p)
    for (k, v), a in zip(prim.items(), holm(ps)):
        v["p_holm"] = round(a, 4)
    o["primary"] = prim
    o["control_beats_all_QAE"] = {c: all(o["means"]["Control"][c] > o["means"][lab][c] for lab, _, _ in G[:5]) for c in CONDS}
    qs = [r for r in ok if r["n"] == n and r["arm"] != "fixed"]
    drop = [mean(r, BASE) - mean(r, PRIM) for r in qs]
    o["drop_vs_cnot_spearman"] = round(float(stats.spearmanr(drop, [r["f2"] for r in qs])[0]), 3)
    o["drop_range"] = [round(min(drop), 4), round(max(drop), 4)]
    out[f"n{n}"] = o
# repro: group means under the unchanged main model against rev2_numbers.json (rounded to 3 decimals there)
rv = json.load(open(os.path.join(ROOT, "results/revision2/rev2_numbers.json"))); rep = {}
for n in (4, 6):
    for lab in ("NA cheap", "NA ideal", "NA maxfid", "MI cheap", "MI ideal"):
        rep[f"n{n} {lab}"] = [out[f"n{n}"]["means"][lab]["fresh|main"], rv[f"n{n} fresh_main"][lab.replace("MI", "MI")][0]]
    rep[f"n{n} Control"] = [out[f"n{n}"]["means"]["Control"]["fresh|main"], rv[f"fixed n{n}/fixed"]["fresh_main"][0]]
out["repro_vs_rev2_fresh_main"] = rep
out["repro_max_abs_diff"] = round(max(abs(a - b) for a, b in rep.values()), 4)
json.dump(out, open(os.path.join(ROOT, "results/hwreal/hwreal_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
