# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Summarise the final evaluation (test, opened once): reproduction, test fidelity per arm and condition, floor,
paired NA-MOPSO vs each arm on the main condition (Wilcoxon-Holm, A12, same run seed). Output results/final_v2/final_report.json."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae.analysis import wilcoxon_vs
ARMS = ["namopso", "mopso_ideal", "nsga2", "moead", "random", "pso", "hho", "bhho"]
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "results", "final_v2")
recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(R, "out", "*.jsonl"))) for l in open(f)]
assert len({r["key"] for r in recs}) == len(recs) and not any(r["smoke"] for r in recs)
out = {"n_records": len(recs), "reproduced": sum(r["status"] == "ok" for r in recs),
       "max_repro_diff": max(r["repro_abs_diff"] for r in recs), "per_n": {}}
for n in sorted({r["n"] for r in recs}):
    rr = {(r["arm"] if "arm" in r else r["key"].split("/")[1], r["run_seed"]): r for r in recs if r["n"] == n and r["status"] == "ok"}
    conds = list(next(iter(rr.values()))["test"])
    seeds = sorted({s for _, s in rr})
    tab = {a: {c: [rr[(a, s)]["test"][c] for s in seeds if (a, s) in rr] for c in conds} for a in ARMS}
    res = {"floor": next(iter(rr.values()))["floor"], "conditions": conds,
           "mean": {a: {c: float(np.mean(v)) for c, v in t.items()} for a, t in tab.items()},
           "sd": {a: {c: float(np.std(v, ddof=1)) for c, v in t.items()} for a, t in tab.items()},
           "f2": {a: float(np.mean([rr[(a, s)]["f2"] for s in seeds])) for a in ARMS},
           "above_floor_main": {a: int(sum(x > res_f for x in tab[a]["main"])) for a in ARMS for res_f in [next(iter(rr.values()))["floor"]["p_in=0.2"]]}}
    main = {a: np.array(tab[a]["main"]) for a in ARMS}
    res["wilcoxon_main_namopso_vs"] = wilcoxon_vs(main["namopso"], {a: main[a] for a in ARMS if a != "namopso"})
    p05 = {a: np.array(tab[a]["p2=0.05"]) for a in ARMS}
    res["wilcoxon_p2_0.05_namopso_vs"] = wilcoxon_vs(p05["namopso"], {a: p05[a] for a in ARMS if a != "namopso"})
    res["ideal_vs_main_gap"] = {a: float(np.mean(np.array(tab[a]["ideal"]) - main[a])) for a in ARMS}
    out["per_n"][str(n)] = res
json.dump(out, open(os.path.join(R, "final_report.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_records", "reproduced", "max_repro_diff")}))
for n, r in out["per_n"].items():
    print(f"\n== n={n} floor {r['floor']}")
    hdr = ["ideal", "main", "p2=0.005", "p2=0.02", "p2=0.05", "T1=T2=50us", "reset=0.05", "p_in=0.1", "p_in=0.3"]
    print(f"{'arm':12s} f2   " + " ".join(f"{h:>10s}" for h in hdr) + "  >floor")
    for a in ARMS:
        print(f"{a:12s} {r['f2'][a]:4.1f} " + " ".join(f"{r['mean'][a][h]:10.4f}" for h in hdr) + f"  {r['above_floor_main'][a]}/10")
    for a, w in r["wilcoxon_main_namopso_vs"].items():
        print(f"  main NA vs {a:11s} p {w['p']:.4f} holm {w['p_holm']:.4f} A12 {w['a12']:.2f}")
