# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Summarise the capacity audit of the zero-latent (fixed-output) control (results/audit_capacity/out)."""
import json, glob, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, "results/audit_capacity/out/*.jsonl"))) for l in open(f)]
assert len(R) == 72 and not any(r["dry"] for r in R) and len({r["key"] for r in R}) == 72
out = {}
for n in (4, 6):
    v = [r for r in R if r["n"] == n]; opt = v[0]["optimum"]
    bi = max(v, key=lambda r: r["val_ideal"]); bn = max(v, key=lambda r: r["val_noisy"])
    byd = {}
    for d in range(1, 7):
        w = [r for r in v if f"/p{d}/" in r["key"]]
        byd[d] = {"best_val_ideal": max(r["val_ideal"] for r in w), "best_val_noisy": max(r["val_noisy"] for r in w),
                  "cnot_range": [min(r["prep_cnot"] for r in w), max(r["prep_cnot"] for r in w)]}
    out[n] = {"optimum_val": opt["val"], "optimum_test": opt["test"],
              "best_ideal": {"key": bi["key"], "val": bi["val_ideal"], "test": bi["test_ideal"], "gap_to_opt_val": opt["val"] - bi["val_ideal"]},
              "best_noisy": {"key": bn["key"], "val_noisy": bn["val_noisy"], "cnot": bn["prep_cnot"]},
              "n_within_0.02_of_opt": sum(opt["val"] - r["val_ideal"] <= 0.02 for r in v), "by_depth": byd}
json.dump(out, open(os.path.join(ROOT, "results/audit_capacity/capacity_report.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
