# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""G0-d: NA-MOPSO is a sane multi-objective sampler on data-free benchmarks (optimum at centre and at random
shifts in the same run), compared with pymoo NSGA-II and random search by hypervolume over 10 seeds.
Pass rule (the benchmark gate): median HV(MOPSO) > median HV(random) on every problem at budget 500. No win over
NSGA-II is required; the numbers are written to results/g0/g0d_hv.json."""
import json, os
import numpy as np
from qmiqcae.search import algorithms as A

DIM = 7

def zdt1(shift):
    def f(x):
        z = np.abs(x - shift); z[0] = x[0]
        g = 1 + 9 * z[1:].mean()
        return [x[0], g * (1 - np.sqrt(x[0] / g))]
    return f

def zdt2(shift):
    def f(x):
        z = np.abs(x - shift); z[0] = x[0]
        g = 1 + 9 * z[1:].mean()
        return [x[0], g * (1 - (x[0] / g) ** 2)]
    return f

def dtlz2(shift):
    def f(x):
        g = np.sum((x[1:] - shift[1:]) ** 2)
        return [(1 + g) * np.cos(x[0] * np.pi / 2), (1 + g) * np.sin(x[0] * np.pi / 2)]
    return f

def hv2(F, ref):
    F = np.asarray(F); F = F[np.all(F < ref, 1)]
    if len(F) == 0: return 0.0
    F = F[A.nondominated_mask(F)]; F = F[np.argsort(F[:, 0])]
    hv, p = 0.0, ref[1]
    for a, b in F:
        if b < p: hv += (ref[0] - a) * (p - b); p = b
    return hv

def run(problem_maker, method, seed, budget, shift):
    ev = A.BenchEvaluator(problem_maker(shift), budget)
    rng = np.random.default_rng(seed)
    if method == "mopso": A.mopso(ev, DIM, rng, budget_hint=budget)
    elif method == "nsga2": A.nsga2(ev, DIM, seed)
    else: A.random_search(ev, DIM, rng)
    return ev.F

def test_g0d():
    budget, out, ok = 500, {}, True
    problems = {"zdt1": (zdt1, [1.1, 11.0]), "zdt2": (zdt2, [1.1, 11.0]), "dtlz2": (dtlz2, [2.5, 2.5])}
    for pname, (mk, ref) in problems.items():
        for shift_kind in ("centre", "random"):
            hv = {m: [] for m in ("mopso", "nsga2", "random")}
            for s in range(10):
                shift = np.full(DIM, 0.5) if shift_kind == "centre" else np.random.default_rng(1000 + s).uniform(0.1, 0.9, DIM)
                for m in hv:
                    hv[m].append(hv2(run(mk, m, s, budget, shift), np.array(ref)))
            med = {m: float(np.median(v)) for m, v in hv.items()}
            out[f"{pname}/{shift_kind}"] = {"median_hv": med, "all": hv}
            ok &= med["mopso"] > med["random"]
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "results", "g0"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "g0", "g0d_hv.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    assert ok, {k: v["median_hv"] for k, v in out.items()}
