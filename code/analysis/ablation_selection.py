# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Post hoc ablation of the selection rule (added after the single test opening; reported as post hoc).
Validation only. For every multi-objective run, the front is built on the arm's own search objective (as in the main
analysis) and three rules pick one point:
  R0 cheap   = F4.4 pre-registered rule: lowest f1 among front points with f2 <= median f2 of the front
  R1 maxfid  = lowest f1 on the front (ties: lowest f2)
  R2 ideal   = front point closest (Euclidean) to the ideal point (min f1, min f2) in (f1, f2/f2_max) space
Writes results/ablation/selection_val.json and results/ablation/to_test.json (only picks that differ from R0, in the
analysis.json format read by experiments/final_eval_v2.py)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import qae as Q
from qmiqcae.analysis import select_pareto_point
from qmiqcae.search.algorithms import nondominated_mask
from analyze_search_v2 import load, objectives
MO = ["namopso", "mopso_ideal", "nsga2", "moead", "random"]


def pick_maxfid(F):
    nd = np.where(nondominated_mask(F))[0]
    return int(sorted(nd, key=lambda i: (F[i, 0], F[i, 1]))[0])


def pick_ideal(F, fm):
    nd = np.where(nondominated_mask(F))[0]
    G = np.column_stack([F[:, 0], F[:, 1] / fm])
    z = G[nd].min(0)
    d = np.sqrt(((G[nd] - z) ** 2).sum(1))
    order = sorted(range(len(nd)), key=lambda j: (round(d[j], 12), F[nd[j], 0]))
    return int(nd[order[0]])


if __name__ == "__main__":
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
    runs, evals = load(os.path.join(R, "search_v2"))
    A = json.load(open(os.path.join(R, "search_v2", "analysis.json")))
    val, test = {}, {"per_n": {}}
    for k in sorted(runs):
        n, arm, s = k
        if arm not in MO:
            continue
        cond = "ideal" if arm == "mopso_ideal" else "noisy"
        F = objectives(evals[k], cond); fm = Q.f2_max_q(n)
        picks = {"cheap": select_pareto_point(F), "maxfid": pick_maxfid(F), "ideal": pick_ideal(F, fm)}
        ref = A["per_n"][str(n)]["selected"][f"{arm}/s{s}"]
        assert evals[k][picks["cheap"]]["eval"] == ref["eval"], k  # R0 reproduces the frozen selection
        for rule, i in picks.items():
            r = evals[k][i]
            item = {"eval": r["eval"], "run_seed": s, "phenotype": r["phenotype"], "f2": r["f2"],
                    "val_fid_noisy": r["val_fid_noisy"], "val_fid_ideal": r["val_fid_ideal"], "selected_on": cond,
                    "rule": rule, "same_as_cheap": r["eval"] == ref["eval"]}
            val[f"n{n}/{arm}/s{s}/{rule}"] = item
            if rule != "cheap" and not item["same_as_cheap"]:
                test["per_n"].setdefault(str(n), {"selected": {}})["selected"][f"{arm}/s{s}@{rule}"] = item
    os.makedirs(os.path.join(R, "ablation"), exist_ok=True)
    json.dump(val, open(os.path.join(R, "ablation", "selection_val.json"), "w"), indent=1)
    # keep distinct (arm, seed, eval) only once in the test list
    for n, d in test["per_n"].items():
        seen, keep = set(), {}
        for key, it in sorted(d["selected"].items()):
            u = (key.split("@")[0], it["eval"])
            if u not in seen:
                seen.add(u); keep[key] = it
        d["selected"] = keep
    json.dump(test, open(os.path.join(R, "ablation", "to_test.json"), "w"), indent=1)
    for n in ("4", "6"):
        print("n", n, "models to test:", len(test["per_n"].get(n, {}).get("selected", {})))
        for arm in MO:
            row = []
            for rule in ("cheap", "maxfid", "ideal"):
                its = [v for kk, v in val.items() if kk.startswith(f"n{n}/{arm}/") and kk.endswith("/" + rule)]
                row.append(f"{rule}: f2 {np.mean([x['f2'] for x in its]):5.1f} noisyval {np.mean([x['val_fid_noisy'] for x in its]):.3f}")
            print(f"  {arm:12s}", " | ".join(row))
