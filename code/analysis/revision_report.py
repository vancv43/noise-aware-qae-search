# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Analysis of the revision kernels (results/revision/qmi-qae-rev-*/out). Prints every number used in the manuscript
and writes results/revision/revision_numbers.json."""
import glob, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import qae as Q
from qmiqcae.analysis import a12, holm, igd_plus, hypervolume
from qmiqcae.search.algorithms import nondominated_mask
from analyze_search_v2 import load, objectives
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
def rd(task):
    return [json.loads(l) for f in glob.glob(f"{R}/revision/*/out/{task}_shard*.jsonl") for l in open(f) if not json.loads(l).get("dry")]
rng = np.random.default_rng(0); OUT = {}
def show(k, v):
    OUT[k] = v; print(k, v)

# ---------------------------------------------------------------- enum: exhaustive grid
E = rd("enum"); show("enum records", len(E))
runs, evals = load(os.path.join(R, "search_v2"))
ARMS = ["random", "pso", "hho", "bhho", "nsga2", "moead", "mopso_ideal", "namopso"]
for n in (4, 6):
    G = [e for e in E if e["n"] == n]; fm = Q.f2_max_q(n)
    F = np.array([[1 - e["val_fid_noisy"], e["f2"]] for e in G]); Z = F[nondominated_mask(F)]
    show(f"grid n{n} size", len(G))
    show(f"grid n{n} front", sorted([[round(1 - a, 3), int(b)] for a, b in Z], key=lambda x: x[1]))
    show(f"grid n{n} best noisy val", round(float(1 - F[:, 0].min()), 3))
    show(f"grid n{n} HV", round(hypervolume(F, f2_max=fm), 4))
    for arm in ARMS:
        hv = [hypervolume(objectives(evals[(n, arm, s)]), f2_max=fm) for s in range(10)]
        ig = [igd_plus(objectives(evals[(n, arm, s)]), Z, f2_max=fm) for s in range(10)]
        best = [1 - objectives(evals[(n, arm, s)])[:, 0].min() for s in range(10)]
        show(f"n{n} {arm} HV/gridHV median, IGD+ to grid front median, best noisy val mean",
             [round(float(np.median(hv)) / hypervolume(F, f2_max=fm), 4), round(float(np.median(ig)), 4), round(float(np.mean(best)), 3)])
    # Kendall tau on the uniform grid, bootstrap clustered by structure (lr variants resampled together)
    a = np.array([e["val_fid_ideal"] for e in G]); b = np.array([e["val_fid_noisy"] for e in G])
    struct = np.array([e["key"].rsplit("/lr", 1)[0] for e in G]); us = np.unique(struct)
    idx = {u: np.where(struct == u)[0] for u in us}
    bs = []
    for _ in range(2000):
        pick = np.concatenate([idx[u] for u in rng.choice(us, len(us))]); bs.append(stats.kendalltau(a[pick], b[pick])[0])
    show(f"grid n{n} kendall tau, cluster CI", [round(float(stats.kendalltau(a, b)[0]), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)])
    # rank of the ideal-best grid circuit under noise
    io = np.argsort(-a); show(f"grid n{n} noisy rank of ideal-best circuit", int(np.where(np.argsort(-b) == io[0])[0][0]) + 1)

# ---------------------------------------------------------------- retest: per-state CIs, OOD, Sherbrooke
T = {r["key"]: r for r in rd("retest")}
show("retest records", len(T)); show("retest not ok", [k for k, v in T.items() if v.get("status") != "ok"])
show("retest max repro", max(v["repro_abs_diff"] for v in T.values()))
val = json.load(open(f"{R}/ablation/selection_val.json"))
BYEVAL = {}
TT = json.load(open(f"{R}/ablation/to_test.json"))
for nn, r in TT["per_n"].items():
    for key, it in r["selected"].items():
        arm_, rest = key.split("/"); s_ = int(rest.split("@")[0][1:])
        BYEVAL[(int(nn), arm_, s_, it["eval"])] = T[f"ablation/n{nn}/{key}"]
def models(n, arm, rule):
    out = []
    for s in range(10):
        it = val.get(f"n{n}/{arm}/s{s}/{rule}") if rule != "cheap" else None
        if rule == "cheap" or it["same_as_cheap"]:
            out.append(T[f"final/n{n}/{arm}/s{s}"])
        else:
            out.append(BYEVAL[(n, arm, s, it["eval"])])
    return out
GROUPS = [("NA cheap", "namopso", "cheap"), ("NA maxfid", "namopso", "maxfid"), ("NA ideal", "namopso", "ideal"),
          ("ideal cheap", "mopso_ideal", "cheap"), ("ideal ideal", "mopso_ideal", "ideal")]
for n in (4, 6):
    for cond in ("main", "p2=0.05", "sherbrooke", "ood_main", "ideal"):
        row = {}
        for lab, arm, rule in GROUPS:
            M = np.array([m["test"][cond] for m in models(n, arm, rule)])     # (10 runs, states)
            st = M.mean(0); bs = [st[rng.integers(0, len(st), len(st))].mean() for _ in range(2000)]
            row[lab] = [round(float(st.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        show(f"n{n} {cond} mean [state CI]", row)
    fl = models(n, "namopso", "cheap")[0]["floor"]; show(f"n{n} floors", {k: round(v, 3) for k, v in fl.items()})
    # paired tests by seed for the decisive comparisons under Sherbrooke and OOD
    for cond in ("sherbrooke", "ood_main"):
        x = [np.mean(m["test"][cond]) for m in models(n, "namopso", "maxfid")]
        P, A_ = {}, {}
        for lab, arm, rule in (("ideal cheap", "mopso_ideal", "cheap"), ("ideal ideal", "mopso_ideal", "ideal"), ("NA cheap", "namopso", "cheap")):
            y = [np.mean(m["test"][cond]) for m in models(n, arm, rule)]
            P[lab] = stats.wilcoxon(x, y).pvalue; A_[lab] = a12(x, y)
        H = holm(P); show(f"n{n} {cond} NA maxfid vs", {k: (round(H[k], 3), round(A_[k], 2)) for k in P})
    # all 8 methods (pre-declared rule) under Sherbrooke and OOD
    for cond in ("sherbrooke", "ood_main"):
        show(f"n{n} {cond} all methods (cheap rule)", {arm: round(float(np.mean([np.mean(T[f'final/n{n}/{arm}/s{s}']['test'][cond]) for s in range(10)])), 3) for arm in ARMS})

# ---------------------------------------------------------------- fixed HEA baseline
for r in sorted(rd("hea"), key=lambda r: r["key"]):
    row = {"lr": r["lr"], "f2": r["f2"]}
    for cond in ("main", "p2=0.05", "sherbrooke", "ideal"):
        v = [np.mean(x["test"][cond]) for x in r["runs"]]; row[cond] = [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3)]
    show(f"HEA {r['key']}", row)

# ---------------------------------------------------------------- oracle
for r in rd("oracle"):
    show(f"oracle {r['key']}", {"test_ideal": {k: round(v, 3) for k, v in r["test_ideal"].items()}, "ood_ideal": round(r["ood_ideal"], 3), "top2_share": round(r["eigs_top2_share"], 4)})

# ---------------------------------------------------------------- noise-aware training
NT = {r["key"]: r for r in rd("ntrain")}; show("ntrain records", len(NT))
for n in (4, 6):
    for lab, arm, rule, keyf in (("NA maxfid", "namopso", "maxfid", lambda s: f"n{n}/namopso/s{s}/maxfid"),
                                  ("ideal cheap", "mopso_ideal", "cheap", lambda s: f"n{n}/mopso_ideal/s{s}")):
        base = models(n, arm, rule)
        for cond in ("main", "p2=0.05", "sherbrooke"):
            x = [np.mean(NT[keyf(s)]["test"][cond]) for s in range(10)]; y = [np.mean(m["test"][cond]) for m in base]
            p = stats.wilcoxon(x, y).pvalue if np.any(np.array(x) != np.array(y)) else 1.0
            show(f"n{n} {lab} {cond} noisy-trained vs ideal-trained", [round(float(np.mean(x)), 3), round(float(np.mean(y)), 3), round(float(p), 4), round(a12(x, y), 2)])

# ---------------------------------------------------------------- shots (analytic)
for N in (1024, 8192):
    show(f"swap-test SE bound over 64 states, {N} shots/state", round(1 / (8 * np.sqrt(N)), 4))
json.dump(OUT, open(f"{R}/revision/revision_numbers.json", "w"), indent=1, default=float)
