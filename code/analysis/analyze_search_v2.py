# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""C1/C2 analysis of the v2.1 search (validation objectives only; never reads test).
Input: results/search_v2/<kernel>/out/<arm>_s<seed>/{runs.jsonl, evals_*.jsonl}. Output: results/search_v2/analysis.json.
- C1: per run HV and IGD+ of the evaluated set in (f1_noisy, f2/F2_MAX(n)), ref (1, 1 + 1/F2_MAX). Every arm is scored
  on the noisy f1 (recorded for every evaluation), so MOPSO-ideal is judged on the same objective. Reference front for
  IGD+ = non-dominated union of all arms and runs per n. Friedman (blocks = seeds) + Wilcoxon-Holm NA-MOPSO vs each, A12.
- C2: selected point per run (F4.4 on the arm's own search objective) -> listed for final evaluation; Kendall tau between
  ideal and noisy val fidelity over all evaluations (bootstrap 95% CI).
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae import qae as Q
from qmiqcae.analysis import hypervolume, igd_plus, select_pareto_point, friedman, wilcoxon_vs
from qmiqcae.search.algorithms import nondominated_mask
ARMS = ["namopso", "mopso_ideal", "nsga2", "moead", "random", "pso", "hho", "bhho"]


def load(root, include_smoke=False):
    runs, evals = {}, {}
    for f in glob.glob(os.path.join(root, "**", "runs.jsonl"), recursive=True):
        for r in map(json.loads, open(f)):
            if (r.get("smoke") and not include_smoke) or r["status"] != "ok":
                continue
            key = (r["n"], r["arm"], r["seed"])
            ev = os.path.join(os.path.dirname(f), f"evals_n{r['n']}_{r['arm']}_s{r['seed']}.jsonl")
            recs = [json.loads(l) for l in open(ev)]
            assert len({x["eval"] for x in recs}) == r["problem"]["unique"] == len(recs), key  # budget exact, no dupes
            runs[key], evals[key] = r, recs
    return runs, evals


def objectives(recs, condition="noisy"):
    v = "val_fid_noisy" if condition == "noisy" else "val_fid_ideal"
    return np.array([[1 - x[v] if x["status"] == "ok" else 1.0, x["f2"]] for x in recs], float)


def analyze(root, n_boot=2000, seed=0, include_smoke=False):
    runs, evals = load(root, include_smoke)
    out = {"n_runs": len(runs), "per_n": {}}
    rng = np.random.default_rng(seed)
    for n in sorted({k[0] for k in runs}):
        fm = Q.f2_max_q(n)
        keys = [k for k in runs if k[0] == n]
        union = np.vstack([objectives(evals[k]) for k in keys])
        Z = union[nondominated_mask(union)]
        seeds = sorted({k[2] for k in keys})
        arms = [a for a in ARMS if all((n, a, s) in runs for s in seeds)]
        hv = {a: np.array([hypervolume(objectives(evals[(n, a, s)]), f2_max=fm) for s in seeds]) for a in arms}
        igd = {a: np.array([igd_plus(objectives(evals[(n, a, s)]), Z, f2_max=fm) for s in seeds]) for a in arms}
        res = {"seeds": seeds, "arms": arms, "f2_max": fm, "ref_front": Z.tolist(),
               "hv": {a: {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                          "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()} for a, v in hv.items()},
               "igd_plus": {a: {"median": float(np.median(v)), "mean": float(v.mean())} for a, v in igd.items()}}
        if len(seeds) >= 5 and len(arms) >= 3:
            res["friedman_hv"] = friedman(np.column_stack([hv[a] for a in arms]))
            res["wilcoxon_hv_namopso_vs"] = wilcoxon_vs(hv["namopso"], {a: hv[a] for a in arms if a != "namopso"})
            res["wilcoxon_igd_namopso_vs"] = wilcoxon_vs(-igd["namopso"], {a: -igd[a] for a in arms if a != "namopso"})
        sel = {}
        for k in keys:
            cond = "ideal" if k[1] == "mopso_ideal" else "noisy"
            i = select_pareto_point(objectives(evals[k], cond))
            r = evals[k][i]
            sel[f"{k[1]}/s{k[2]}"] = {"eval": r["eval"], "run_seed": k[2], "phenotype": r["phenotype"], "f2": r["f2"],
                                      "val_fid_noisy": r["val_fid_noisy"], "val_fid_ideal": r["val_fid_ideal"],
                                      "selected_on": cond}
        res["selected"] = sel
        allr = [x for k in keys for x in evals[k] if x["status"] == "ok"]
        a_ = np.array([x["val_fid_ideal"] for x in allr]); b_ = np.array([x["val_fid_noisy"] for x in allr])
        tau = stats.kendalltau(a_, b_)[0]
        bs = [stats.kendalltau(a_[i], b_[i])[0] for i in (rng.integers(0, len(a_), len(a_)) for _ in range(n_boot))]
        res["kendall_ideal_vs_noisy"] = {"tau": float(tau), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         "n_evals": len(allr)}
        out["per_n"][str(n)] = res
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "results", "search_v2")
    o = analyze(root, n_boot=int(os.environ.get("NBOOT", "2000")))
    json.dump(o, open(os.path.join(root, "analysis.json"), "w"), indent=1)
    for n, r in o["per_n"].items():
        print("n", n, {a: round(v["median"], 4) for a, v in r["hv"].items()}, r.get("friedman_hv"))
