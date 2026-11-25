# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Revision experiments, pre-specified in results/revision2/PRESPEC.md.
Tasks (one jsonl line per item, resumable, sharded):
  confirm : the 287 selections of roadmap_v1 'retest', retrained with their original seeds (repro check), then
            fresh in-distribution states (128, seed 15) under main, joint p2=0.05, Sherbrooke; and on the original test
            states the isolated sweeps p2 in {0.005,0.02,0.05} at p1=1e-3 and p1 in {1e-4,5e-3} at p2=0.01.
  seeds   : swarm-variant selections (NA cheap/ideal/maxfid, MOPSO-ideal cheap/ideal), 3 fresh training seeds each,
            tested on the original test and the fresh states under main and joint p2=0.05.
  fixed   : zero-latent-qubit control (every qubit reset, so the decoder prepares one trained state);
            topology x depth 1-3 x lr on NOISY validation at seed 0, best one retrained with 10 seeds and tested.
            Also the ideal constant-state reference: leading eigenvector of the mean clean TRAINING state.
Usage: python roadmap_v2.py TASK OUTDIR SEL_FINAL SEL_ABL SEL_VAL [--shard I N] [--dry]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmiqcae import sim, qae as Q
import roadmap_v1 as R1

MAIN, P205, SHER = R1.MAIN, R1.P205, R1.SHERBROOKE
ISO = {"p2only=0.005": sim.NoiseModel(p1=1e-3, p2=0.005), "p2only=0.02": sim.NoiseModel(p1=1e-3, p2=0.02),
       "p2only=0.05": sim.NoiseModel(p1=1e-3, p2=0.05), "p1only=1e-4": sim.NoiseModel(p1=1e-4, p2=0.01),
       "p1only=5e-3": sim.NoiseModel(p1=5e-3, p2=0.01)}
FRESH_SEED, FRESH_N = 15, 128
NEW_SEEDS = (7001, 7002, 7003)
GROUPS = {("namopso", "cheap"), ("namopso", "ideal"), ("namopso", "maxfid"), ("mopso_ideal", "cheap"), ("mopso_ideal", "ideal")}


def fresh_states(n):
    h = np.sort(np.random.default_rng(FRESH_SEED).uniform(*Q.H_RANGE, FRESH_N))
    sp = Q.QSplits(n, mode="final"); used = np.concatenate(list(sp.h.values()))
    assert np.min(np.abs(h[:, None] - used[None, :])) > 1e-9
    return torch.tensor(np.stack([Q.tfim_ground(n, x) for x in h]), dtype=torch.complex128)


def evaluate(m, n, sp, iso=True):
    te, fr = sp.states("test"), fresh_states(n)
    out = {"fresh_main": R1.per_state(m, fr, MAIN), "fresh_p2=0.05": R1.per_state(m, fr, P205),
           "fresh_sherbrooke": R1.per_state(m, fr, SHER), "test_main": R1.per_state(m, te, MAIN)}
    if iso:
        out.update({k: R1.per_state(m, te, v) for k, v in ISO.items()})
    return out, {"test": R1.floor(te), "fresh": R1.floor(fr)}


# ---------------------------------------------------------------- zero-latent model (decoder prepares a fixed state)
def forward0(spec, params, rho_in, noise):
    n = spec.n
    if noise is None:
        return Q._qae_ideal_unitary(spec, 0, params, rho_in)          # dl = 1: Tr over all qubits, exact
    S1, S2 = sim.noise_superops(noise); gates = Q.qae_gates(spec)
    rho = Q._apply_gates(rho_in.to(sim.CDTYPE), gates, params, S1, S2, inverse=False)
    for q in range(n):
        rho = Q.reset_qubit(rho, q, noise.e0)
    return Q._apply_gates(rho, gates, params, S1, S2, inverse=True)


class QAE0(Q.QAE):
    def __init__(self, spec, seed):
        super().__init__(spec, 1, seed=seed); self.k = 0

    def forward(self, rho_in, noise=None):
        return forward0(self.spec, self.theta, rho_in, noise)


def train0(spec, lr, seed, epochs, sp):
    m = QAE0(spec, seed)
    info = Q.train_qae(m, sp.states("train"), sp.states("val"),
                       Q.QTrainCfg(lr_max=lr, epochs=epochs, patience=epochs + 1, p_in=0.2, seed=seed))
    return m, info


def items(task, SF, SA):
    base = R1.items("retest", SF, SA)
    if task == "confirm":
        return base
    if task == "seeds":
        out = []
        for it in base:
            key = it["key"]; arm = key.split("/")[2]; rule = key.split("@")[1] if "@" in key else "cheap"
            if (arm, rule) in GROUPS:
                out += [{**it, "key": f"{key}#seed{s}", "seed": s} for s in NEW_SEEDS]
        return out
    if task == "fixed":
        return [{"key": f"n{n}/fixed", "n": n} for n in (4, 6)]
    raise ValueError(task)


def run(task, it, epochs):
    n = it["n"]; sp = Q.QSplits(n, mode="final")
    if task in ("confirm", "seeds"):
        s = it["sel"]; ph = Q.QPhenotype(**s["phenotype"])
        seed = it.get("seed", s["run_seed"] * 100003 + s["eval"])
        m, info = R1.train(ph, seed, epochs, sp)
        rep = abs(info["best_val_fid"] - s["val_fid_ideal"])
        if task == "confirm" and epochs == 100 and rep > R1.REPRO_TOL:
            return {"status": "not_reproduced", "repro_abs_diff": rep}
        va = float(Q.eval_fidelity(m, sp.states("val"), 0.2, noise=MAIN).mean())
        t, fl = evaluate(m, n, sp, iso=(task == "confirm"))
        return {"status": "ok", "repro_abs_diff": rep if task == "confirm" else None, "f2": s["f2"], "k": ph.k,
                "val_fid_noisy": va, "val_floor": R1.floor(sp.states("val")), "test": t, "floor": fl}
    if task == "fixed":
        va = sp.states("val"); scores = {}
        for t in ("ring", "linear", "all"):
            for d in (1, 2, 3):
                for lr in R1.LRS:
                    spec = Q.QPhenotype(n=n, topology=t, depth=d, rotset="XYZ", k=1, lr=lr).spec
                    m, _ = train0(spec, lr, 0, epochs, sp)
                    scores[f"{t}/p{d}/lr{lr}"] = float(Q.eval_fidelity(m, va, 0.2, noise=MAIN).mean())
        best = max(scores, key=scores.get); t, d, lr = best.split("/"); d = int(d[1:]); lr = float(lr[2:])
        spec = Q.QPhenotype(n=n, topology=t, depth=d, rotset="XYZ", k=1, lr=lr).spec
        runs = []
        for s in range(10):
            m, info = train0(spec, lr, 1000 + s, epochs, sp); tt, fl = evaluate(m, n, sp, iso=False)
            runs.append({"seed": 1000 + s, "val_fid_ideal": info["best_val_fid"],
                         "test_ideal": R1.per_state(m, sp.states("test"), None), "test": tt, "floor": fl})
        tr = sp.states("train"); rho = torch.einsum("bi,bj->ij", tr, tr.conj()) / len(tr)
        w, v = torch.linalg.eigh(rho); phi = v[:, -1]
        const = {s: float((torch.abs(X.conj() @ phi) ** 2).mean()) for s, X in
                 (("test", sp.states("test")), ("fresh", fresh_states(n)))}
        return {"scores": scores, "best": best, "cnot": Q.n_cnot_qae(spec), "runs": runs,
                "ideal_constant_state": const, "train_spectrum_top3": [float(x) for x in w.flip(0)[:3].real]}


if __name__ == "__main__":
    task, outdir, sf, sa, sv = sys.argv[1:6]; dry = "--dry" in sys.argv
    torch.set_num_threads(1); sim.set_precision("c64"); os.makedirs(outdir, exist_ok=True)
    si, sn = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
    SF, SA = json.load(open(sf)), json.load(open(sa)); R1.SV.update(json.load(open(sv)))
    OUT = os.path.join(outdir, f"{task}_shard{si}.jsonl")
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    its = items(task, SF, SA)
    if dry:
        its = its[:: max(1, len(its) // 2)][:2]
    for j, it in enumerate(its):
        if j % sn != si or it["key"] in done:
            continue
        t0 = time.time(); res = run(task, it, 2 if dry else 100)
        open(OUT, "a").write(json.dumps({"key": it["key"], "n": it["n"], **res, "seconds": time.time() - t0, "dry": dry}) + "\n")
        print(task, it["key"], res.get("status", "ok"), round(time.time() - t0, 1), flush=True)
