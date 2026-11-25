# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Hardware-realism re-scoring (pre-specified in results/hwreal/PRESPEC.md).
Extends the main noise model with (i) relaxation of idle qubits under an as-soon-as-possible schedule and (ii) a finite
reset duration during which the latent qubits relax. The selected models are retrained exactly as before (repro check)
and scored on the original test states and on the 128 new states of roadmap_v2.
Usage: python hwreal.py OUTDIR SEL_FINAL SEL_ABL SEL_VAL [--shard I N] [--dry]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import sim, qae as Q
import roadmap_v1 as R1, roadmap_v2 as R2

MAIN, SHER = R1.MAIN, R1.SHERBROOKE
# condition -> (noise model, idle relaxation on?, reset duration in s)
CONDS = {"main": (MAIN, False, 0.0), "main+idle": (MAIN, True, 0.0), "main+idle+reset1us": (MAIN, True, 1e-6),
         "main+idle+reset4us": (MAIN, True, 4e-6), "sher+idle+reset1us": (SHER, True, 1e-6)}
FIXED_CFG = ("linear", 2, 0.1)        # rev2 fixed-output choice for both n (results/revision2/rev2_numbers.json)
FIXED_SEEDS = tuple(range(1000, 1010))
_TH = {}


def thermal_superop(noise, tau):
    key = (noise.t1, noise.t2, round(tau * 1e12))
    if key not in _TH:
        _TH[key] = sim._superop_from_channel(lambda r: sim.dm_thermal(r, noise.t1, noise.t2, tau, 0), 1)
    return _TH[key]


def schedule(gates, n, noise):
    """ASAP moments. Returns a list of (duration, [gate indices]); a gate starts when all its qubits are free."""
    free = [0] * n; moments = []
    for i, g in enumerate(gates):
        qs = (g[1], g[2]) if g[0] == "CNOT" else (g[4],)
        m = max(free[q] for q in qs)
        while len(moments) <= m:
            moments.append([])
        moments[m].append(i)
        for q in qs:
            free[q] = m + 1
    out = []
    for mo in moments:
        dur = max(noise.t_2q if gates[i][0] == "CNOT" else noise.t_1q for i in mo)
        out.append((dur, mo))
    return out


def _apply_scheduled(rho, gates, params, noise, S1, S2, inverse, idle):
    n = sim._dm_n(rho)
    seq = list(reversed(gates)) if inverse else list(gates)
    for dur, mo in schedule(seq, n, noise):
        busy = {}
        for i in mo:
            g = seq[i]
            if g[0] == "CNOT":
                rho = sim.dm_apply_cnot(rho, g[1], g[2]); rho = sim.dm_channel_2q(rho, S2, g[1], g[2])
                busy[g[1]] = busy[g[2]] = noise.t_2q
            else:
                _, ax, layer, ax_i, q = g
                th = params[layer, ax_i, q]
                rho = sim.dm_apply_1q(rho, sim.GATE_FN[ax](-th if inverse else th), q); rho = sim.dm_channel_1q(rho, S1, q)
                busy[q] = noise.t_1q
        if idle and noise.thermal:
            for q in range(n):
                tau = dur - busy.get(q, 0.0)
                if tau > 1e-15:
                    rho = sim.dm_channel_1q(rho, thermal_superop(noise, tau), q)
    return rho


def forward_ext(spec, k, params, rho_in, noise, idle, t_reset):
    """Noisy QAE channel of qae.qae_forward with idle relaxation and a reset duration; k = 0 gives the fixed-output control."""
    n = spec.n; S1, S2 = sim.noise_superops(noise); gates = Q.qae_gates(spec)
    rho = _apply_scheduled(rho_in.to(sim.CDTYPE), gates, params, noise, S1, S2, False, idle)
    for q in range(k, n):
        rho = Q.reset_qubit(rho, q, noise.e0)
    if t_reset > 0 and noise.thermal:
        for q in range(k):
            rho = sim.dm_channel_1q(rho, thermal_superop(noise, t_reset), q)
    return _apply_scheduled(rho, gates, params, noise, S1, S2, True, idle)


@torch.no_grad()
def score(m, psi, cond, batch=32):
    noise, idle, tr = CONDS[cond]; out = []
    for i in range(0, len(psi), batch):
        p = psi[i:i + batch]
        out.append(Q.fidelity(p, forward_ext(m.spec, m.k, m.theta, Q.input_channel(Q.pure_dm(p), 0.2), noise, idle, tr)))
    return [round(float(x), 7) for x in torch.cat(out)]


def items(SF, SA):
    """One item per distinct trained model: the swarm groups of roadmap_v2 (every rule of every run, from
    selection_val.json) deduplicated by (n, arm, run, evaluation); 'rules' lists the rules that chose it."""
    by = {}
    for key, s in sorted(R1.SV.items()):
        n, arm, run, rule = key.split("/")
        if (arm, rule) in R2.GROUPS:
            k = f"{n}/{arm}/{run}/e{s['eval']}"
            by.setdefault(k, {"key": k, "n": int(n[1:]), "kind": "qae", "arm": arm, "sel": s, "rules": []})["rules"].append(rule)
    out = list(by.values())
    out += [{"key": f"n{n}/fixed#seed{s}", "n": n, "kind": "fixed", "seed": s} for n in (4, 6) for s in FIXED_SEEDS]
    return out


def run(it, epochs):
    n = it["n"]; sp = Q.QSplits(n, mode="final")
    if it["kind"] == "qae":
        s = it["sel"]; ph = Q.QPhenotype(**s["phenotype"]); seed = s["run_seed"] * 100003 + s["eval"]
        m, info = R1.train(ph, seed, epochs, sp)
        rep = abs(info["best_val_fid"] - s["val_fid_ideal"])
        if epochs == 100 and rep > R1.REPRO_TOL:
            return {"status": "not_reproduced", "repro_abs_diff": rep}
        meta = {"arm": it["arm"], "rules": it["rules"], "run_seed": s["run_seed"], "f2": s["f2"], "k": ph.k,
                "depth": ph.depth, "topology": ph.topology, "repro_abs_diff": rep}
    else:
        t, d, lr = FIXED_CFG
        spec = Q.QPhenotype(n=n, topology=t, depth=d, rotset="XYZ", k=1, lr=lr).spec
        m, info = R2.train0(spec, lr, it["seed"], epochs, sp)
        meta = {"arm": "fixed", "rules": [], "seed": it["seed"], "f2": Q.n_cnot_qae(spec) // 2, "k": 0, "depth": d, "topology": t}
    te, fr = sp.states("test"), R2.fresh_states(n)
    res = {f"test|{c}": score(m, te, c) for c in ("main", "main+idle", "main+idle+reset1us")}
    res.update({f"fresh|{c}": score(m, fr, c) for c in CONDS})
    return {"status": "ok", **meta, "scores": res, "floor": {"test": R1.floor(te), "fresh": R1.floor(fr)}}


if __name__ == "__main__":
    outdir, sf, sa, sv = sys.argv[1:5]; dry = "--dry" in sys.argv
    si, sn = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
    torch.set_num_threads(1); sim.set_precision("c64"); os.makedirs(outdir, exist_ok=True)
    SF, SA = json.load(open(sf)), json.load(open(sa)); R1.SV.update(json.load(open(sv)))
    its = items(SF, SA)
    if dry:
        its = [its[0], its[-1]]
    OUT = os.path.join(outdir, f"hwreal_shard{si}.jsonl")
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    for j, it in enumerate(its):
        if j % sn != si or it["key"] in done:
            continue
        t0 = time.time(); res = run(it, 2 if dry else 100)
        open(OUT, "a").write(json.dumps({"key": it["key"], "n": it["n"], **res, "seconds": time.time() - t0, "dry": dry}) + "\n")
        print(it["key"], res["status"], round(time.time() - t0, 1), flush=True)
