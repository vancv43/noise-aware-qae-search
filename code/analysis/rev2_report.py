# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Analysis of the revision kernels (results/revision2), following PRESPEC.md."""
import glob, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae.analysis import a12, holm
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
rd = lambda t: [json.loads(l) for f in glob.glob(f"{R}/revision2/*/out/{t}_shard*.jsonl") for l in open(f)]
rng = np.random.default_rng(0); OUT = {}
def show(k, v): OUT[k] = v; print(k, v)
C = {r["key"]: r for r in rd("confirm")}
show("confirm records", len(C)); show("not ok", [k for k, v in C.items() if v["status"] != "ok"])
show("max repro", max(v["repro_abs_diff"] for v in C.values()))
val = json.load(open(f"{R}/ablation/selection_val.json")); TT = json.load(open(f"{R}/ablation/to_test.json"))
BYEVAL = {}
for nn, r in TT["per_n"].items():
    for key, it in r["selected"].items():
        arm, rest = key.split("/"); BYEVAL[(int(nn), arm, int(rest.split("@")[0][1:]), it["eval"])] = f"ablation/n{nn}/{key}"
def key_of(n, arm, s, rule):
    if rule == "cheap": return f"final/n{n}/{arm}/s{s}"
    it = val[f"n{n}/{arm}/s{s}/{rule}"]
    return f"final/n{n}/{arm}/s{s}" if it["same_as_cheap"] else BYEVAL[(n, arm, s, it["eval"])]
def m(rec, cond): return float(np.mean(rec["test"][cond]))
def ci(states):
    st = np.asarray(states); bs = [st[rng.integers(0, len(st), len(st))].mean() for _ in range(2000)]
    return [round(float(st.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
G = [("NA", "namopso", "cheap"), ("NA", "namopso", "ideal"), ("NA", "namopso", "maxfid"),
     ("MI", "mopso_ideal", "cheap"), ("MI", "mopso_ideal", "ideal"), ("MI", "mopso_ideal", "maxfid")]
for n in (4, 6):
    fl = C[key_of(n, "namopso", 0, "cheap")]["floor"]; show(f"n{n} floors", {k: round(v, 3) for k, v in fl.items()})
    for cond in ("fresh_main", "fresh_p2=0.05", "fresh_sherbrooke"):
        row = {}
        for lab, arm, rule in G:
            recs = [C[key_of(n, arm, s, rule)] for s in range(10)]
            M = np.array([r["test"][cond] for r in recs]).mean(0)
            above = sum(m(r, cond) > fl["fresh"] for r in recs)
            row[f"{lab} {rule}"] = ci(M) + [f"{above}/10"]
        show(f"n{n} {cond}", row)
    # primary confirmatory family
    x = [m(C[key_of(n, "namopso", s, "maxfid")], "fresh_main") for s in range(10)]
    P, A = {}, {}
    for lab, rule in (("H1 vs MI cheap", "cheap"), ("H2 vs MI ideal", "ideal")):
        y = [m(C[key_of(n, "mopso_ideal", s, rule)], "fresh_main") for s in range(10)]
        P[lab] = stats.wilcoxon(x, y).pvalue; A[lab] = a12(x, y)
    H = holm(P); show(f"n{n} PRIMARY fresh main", {k: {"p_raw": round(P[k], 4), "p_holm": round(H[k], 4), "a12": round(A[k], 2)} for k in P})
    for cond in ("fresh_p2=0.05", "fresh_sherbrooke"):
        x2 = [m(C[key_of(n, "namopso", s, "maxfid")], cond) for s in range(10)]
        show(f"n{n} secondary {cond} NA maxfid A12 vs MI cheap/ideal",
             [round(a12(x2, [m(C[key_of(n, "mopso_ideal", s, r)], cond) for s in range(10)]), 2) for r in ("cheap", "ideal")])
    # all 5 MO methods under R1, fresh main and p2=0.05
    show(f"n{n} R1 all MO fresh", {a: [round(float(np.mean([m(C[key_of(n, a, s, 'maxfid')], c) for s in range(10)])), 3) for c in ("fresh_main", "fresh_p2=0.05")]
                                   for a in ("random", "nsga2", "moead", "mopso_ideal", "namopso")})
    # success rate under every rule for all 5 MO, fresh main
    show(f"n{n} above-baseline fresh main", {f"{a}/{r}": sum(m(C[key_of(n, a, s, r)], "fresh_main") > fl["fresh"] for s in range(10))
                                             for a in ("random", "nsga2", "moead", "mopso_ideal", "namopso") for r in ("cheap", "ideal", "maxfid")})
    # R2 fallback on the pre-declared rule for all 8 methods: identity if noisy val <= val floor
    for arm in ("random", "pso", "hho", "bhho", "nsga2", "moead", "mopso_ideal", "namopso"):
        recs = [C[f"final/n{n}/{arm}/s{s}"] for s in range(10)]
        fb = [fl["fresh"] if r["val_fid_noisy"] <= r["val_floor"] else m(r, "fresh_main") for r in recs]
        OUT.setdefault(f"n{n} R2 pre-declared fresh main", {})[arm] = [round(float(np.mean([m(r, 'fresh_main') for r in recs])), 3),
                                                                        round(float(np.mean(fb)), 3), sum(r["val_fid_noisy"] <= r["val_floor"] for r in recs)]
    print(f"n{n} R2 pre-declared fresh main [mean, with fallback, n fallback]", OUT[f"n{n} R2 pre-declared fresh main"])
    # isolated sweeps on test states
    conds = ["test_main", "p2only=0.005", "p2only=0.02", "p2only=0.05", "p1only=1e-4", "p1only=5e-3"]
    for lab, arm, rule in (("NA cheap", "namopso", "cheap"), ("NA maxfid", "namopso", "maxfid"), ("MI cheap", "mopso_ideal", "cheap"), ("MI ideal", "mopso_ideal", "ideal")):
        show(f"n{n} iso {lab}", {c: round(float(np.mean([m(C[key_of(n, arm, s, rule)], c) for s in range(10)])), 3) for c in conds})
    # product circuits (f2 = 0) under p2-only: must not change with p2
    prod = [r for r in C.values() if r["n"] == n and r["f2"] == 0]
    show(f"n{n} product circuits max change p2only 0.005->0.05", round(max(abs(m(r, 'p2only=0.05') - m(r, 'p2only=0.005')) for r in prod), 6))
    # p2-only slope vs CNOT among all 287 models: spearman of loss (0.005->0.05) vs f2
    loss = [m(r, "p2only=0.005") - m(r, "p2only=0.05") for r in C.values() if r["n"] == n]; f2 = [r["f2"] for r in C.values() if r["n"] == n]
    show(f"n{n} p2-only loss vs CNOT spearman", round(float(stats.spearmanr(f2, loss)[0]), 2))
# ---------------------------------------------------------------- training seeds
S = rd("seeds"); show("seed records", len(S))
by = {}
for r in S:
    base = r["key"].split("#")[0]; by.setdefault(base, []).append(r)
def seedmean(key, cond):
    if key in by: return float(np.mean([np.mean(r["test"][cond]) for r in by[key]]))
    return None
for n in (4, 6):
    for lab, arm, rule in (("NA cheap", "namopso", "cheap"), ("NA ideal", "namopso", "ideal"), ("NA maxfid", "namopso", "maxfid"), ("MI cheap", "mopso_ideal", "cheap"), ("MI ideal", "mopso_ideal", "ideal")):
        keys = [key_of(n, arm, s, rule) for s in range(10)]
        orig = [m(C[k], "test_main") for k in keys]
        fresh = [[np.mean(r["test"]["test_main"]) for r in by[k]] for k in keys]
        within = float(np.mean([np.std(f, ddof=1) for f in fresh])); across = float(np.std(orig, ddof=1))
        show(f"n{n} seeds {lab} test_main", {"orig_mean": round(float(np.mean(orig)), 3), "fresh_seed_mean": round(float(np.mean(fresh)), 3),
             "within_config_sd": round(within, 3), "across_run_sd": round(across, 3), "max_abs_dev": round(float(max(abs(np.mean(f) - o) for f, o in zip(fresh, orig))), 3)})
    # does H1/H2 hold with fresh-seed means (test states, main)?
    x = [float(np.mean([np.mean(r["test"]["test_main"]) for r in by[key_of(n, "namopso", s, "maxfid")]])) for s in range(10)]
    for rule in ("cheap", "ideal"):
        y = [float(np.mean([np.mean(r["test"]["test_main"]) for r in by[key_of(n, "mopso_ideal", s, rule)]])) for s in range(10)]
        show(f"n{n} seed-avg NA maxfid vs MI {rule}", {"p_raw": round(float(stats.wilcoxon(x, y).pvalue), 4), "a12": round(a12(x, y), 2), "means": [round(float(np.mean(x)), 3), round(float(np.mean(y)), 3)]})
    xs = [float(np.mean([np.mean(r["test"]["fresh_p2=0.05"]) for r in by[key_of(n, "namopso", s, "maxfid")]])) for s in range(10)]
    ys = [float(np.mean([np.mean(r["test"]["fresh_p2=0.05"]) for r in by[key_of(n, "mopso_ideal", s, "cheap")]])) for s in range(10)]
    show(f"n{n} seed-avg fresh p2=0.05 NA maxfid vs MI cheap", [round(float(np.mean(xs)), 3), round(float(np.mean(ys)), 3), round(a12(xs, ys), 2)])
# ---------------------------------------------------------------- fixed-output control
for r in rd("fixed"):
    rows = {c: [round(float(np.mean([np.mean(x["test"][c]) for x in r["runs"]])), 3), round(float(np.std([np.mean(x["test"][c]) for x in r["runs"]], ddof=1)), 3)]
            for c in ("test_main", "fresh_main", "fresh_p2=0.05", "fresh_sherbrooke")}
    rows["test_ideal"] = round(float(np.mean([np.mean(x["test_ideal"]) for x in r["runs"]])), 3)
    show(f"fixed {r['key']}", {"best": r["best"], "cnot_full": r["cnot"], "top3": [round(x, 5) for x in r["train_spectrum_top3"]],
                               "ideal_const": {k: round(v, 4) for k, v in r["ideal_constant_state"].items()}, **rows,
                               "top_scores": sorted(r["scores"].items(), key=lambda kv: -kv[1])[:3]})
json.dump(OUT, open(f"{R}/revision2/rev2_numbers.json", "w"), indent=1, default=str)
