# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Revision experiments after an internal review (post hoc, all disclosed as such).
Tasks (one jsonl line per item, resumable, sharded):
  enum   : every one of the 96 discrete architectures x lr in {0.01, 0.03, 0.1}, seed fixed, VALIDATION ONLY
           (ideal and noisy val fidelity) -> exhaustive reference front, uniform-sample Kendall tau.
  retest : the 160 frozen selections + 127 ablation selections, retrained exactly as final_eval_v2.py (repro check
           1e-5), then per-state test fidelities under ideal, main, p2=0.05, a calibration snapshot of IBM Sherbrooke,
           and the main noise on out-of-distribution states h in [2, 3].
  hea    : fixed ring HEA, rotset YZ, k=1, depth 1-3; lr chosen on NOISY validation from {0.01, 0.03, 0.1} at seed 0;
           then 10 seeds trained and tested like retest.
  oracle : subspace-rotation oracle (top-2 eigenvectors of the mean clean training state, no training) on test.
  ntrain : noise-aware training (loss and early stopping under the main hardware noise) of the NA-QAS-QAE
           highest-fidelity selections and the MOPSO-ideal pre-declared selections; tested like retest.
Usage: python roadmap_v1.py TASK OUTDIR SEL_FINAL SEL_ABL SEL_VAL [--shard I N] [--dry]"""
import json, math, os, sys, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qmiqcae import sim, qae as Q
from qmiqcae.circuits import TOPOLOGIES

MAIN = sim.NoiseModel()
# IBM Sherbrooke (props 2024-05-27, qiskit-ibm-runtime 0.37 fake backend), medians over qubits/pairs.
# RB error r -> Pauli error p = r (d+1)/d (d = 2 or 4); readout error used as reset error.
SHERBROOKE = sim.NoiseModel(p1=2.1249e-4 * 1.5, p2=7.4828e-3 * 1.25, t1=279.56e-6, t2=197.81e-6,
                            t_1q=56.89e-9, t_2q=533.33e-9, e0=0.0117, e1=0.0117)
P205 = sim.NoiseModel(p1=0.005, p2=0.05)
REPRO_TOL = 1e-5
SV = {}
LRS = (0.01, 0.03, 0.1)
OOD_RANGE, OOD_SEED, OOD_N = (2.0, 3.0), 14, 64


def ood_states(n):
    h = np.sort(np.random.default_rng(OOD_SEED).uniform(*OOD_RANGE, OOD_N))
    return torch.tensor(np.stack([Q.tfim_ground(n, x) for x in h]), dtype=torch.complex128)


def floor(psi, pin=0.2):
    return float(Q.fidelity(psi, Q.input_channel(Q.pure_dm(psi), pin)).mean())


def per_state(m, psi, noise, pin=0.2):
    return [round(float(x), 7) for x in Q.eval_fidelity(m, psi, pin, noise=noise)]


def test_block(m, n, sp):
    te = sp.states("test"); od = ood_states(n)
    out = {"ideal": per_state(m, te, None), "main": per_state(m, te, MAIN), "p2=0.05": per_state(m, te, P205),
           "sherbrooke": per_state(m, te, SHERBROOKE), "ood_main": per_state(m, od, MAIN), "ood_ideal": per_state(m, od, None)}
    fl = {"test": floor(te), "ood": floor(od)}
    return out, fl


def train(ph, seed, epochs, sp):
    m = Q.QAE(ph.spec, ph.k, seed=seed)
    info = Q.train_qae(m, sp.states("train"), sp.states("val"),
                       Q.QTrainCfg(lr_max=ph.lr, epochs=epochs, patience=epochs + 1, p_in=0.2, seed=seed))
    return m, info


def train_noisy(ph, seed, epochs, sp, noise=MAIN):
    """Same as Q.train_qae, but the loss and the early-stopping validation fidelity use the hardware noise."""
    m = Q.QAE(ph.spec, ph.k, seed=seed)
    cfg = Q.QTrainCfg(lr_max=ph.lr, epochs=epochs, patience=epochs + 1, p_in=0.2, seed=seed)
    psi_tr, psi_va = sp.states("train"), sp.states("val")
    g = torch.Generator().manual_seed(cfg.seed * 104729 + 3)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr_max)
    n = len(psi_tr); total = math.ceil(n / cfg.batch) * cfg.epochs
    rho_tr = Q.input_channel(Q.pure_dm(psi_tr), cfg.p_in)
    step, best, best_state, best_ep = 0, -1.0, None, 0
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            for pg in opt.param_groups:
                pg["lr"] = Q._lr(step, total, cfg)
            loss = 1 - Q.fidelity(psi_tr[idx], m(rho_tr[idx], noise)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
        vf = float(Q.eval_fidelity(m, psi_va, cfg.p_in, noise=noise).mean())
        if vf > best + 1e-7:
            best, best_ep, best_state = vf, ep + 1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, {"best_val_fid_noisy": best, "best_epoch": best_ep}


def items(task, sel_final, sel_abl):
    if task == "enum":
        return [{"key": f"n{n}/{t}/p{d}/{r}/k{k}/lr{lr}", "n": n, "ph": dict(n=n, topology=t, depth=d, rotset=r, k=k, lr=lr)}
                for n in (4, 6) for t in TOPOLOGIES for d in Q.Q_DEPTHS for r in Q.Q_ROTS for k in Q.Q_LATENT for lr in LRS]
    if task == "retest":
        out = []
        for src, A in (("final", sel_final), ("ablation", sel_abl)):
            for n, r in sorted(A["per_n"].items()):
                for key, s in sorted(r["selected"].items()):
                    out.append({"key": f"{src}/n{n}/{key}", "n": int(n), "sel": s})
        return out
    if task == "hea":
        return [{"key": f"n{n}/ring/p{d}", "n": n, "depth": d} for n in (4, 6) for d in (1, 2, 3)]
    if task == "oracle":
        return [{"key": f"n{n}/oracle", "n": n} for n in (4, 6)]
    if task == "ntrain":
        out = []
        for key, s in sorted(SV.items()):          # selection_val.json: every rule for every run
            if "/namopso/" in key and key.endswith("/maxfid"):
                out.append({"key": key, "n": int(key[1]), "sel": s})
        for n, r in sorted(sel_final["per_n"].items()):
            for key, s in sorted(r["selected"].items()):
                if key.startswith("mopso_ideal/"):
                    out.append({"key": f"n{n}/{key}", "n": int(n), "sel": s})
        return out
    raise ValueError(task)


def run(task, it, epochs, sel_abl_val):
    n = it["n"]; sp = Q.QSplits(n, mode="final")
    if task == "enum":
        ph = Q.QPhenotype(**it["ph"]); m, info = train(ph, 0, epochs, sp)
        va = sp.states("val")
        return {"val_fid_ideal": info["best_val_fid"], "val_fid_noisy": float(Q.eval_fidelity(m, va, 0.2, noise=MAIN).mean()),
                "f2": Q.f2_q(ph), "best_epoch": info["best_epoch"]}
    if task == "retest":
        s = it["sel"]; ph = Q.QPhenotype(**s["phenotype"]); seed = s["run_seed"] * 100003 + s["eval"]
        m, info = train(ph, seed, epochs, sp)
        rep = abs(info["best_val_fid"] - s["val_fid_ideal"])
        if epochs == 100 and rep > REPRO_TOL:
            return {"status": "not_reproduced", "repro_abs_diff": rep}
        t, fl = test_block(m, n, sp)
        return {"status": "ok", "repro_abs_diff": rep, "f2": s["f2"], "test": t, "floor": fl}
    if task == "hea":
        va = sp.states("val"); scores = {}
        for lr in LRS:
            ph = Q.QPhenotype(n=n, topology="ring", depth=it["depth"], rotset="YZ", k=1, lr=lr)
            m, _ = train(ph, 0, epochs, sp); scores[lr] = float(Q.eval_fidelity(m, va, 0.2, noise=MAIN).mean())
        lr = max(scores, key=scores.get)
        ph = Q.QPhenotype(n=n, topology="ring", depth=it["depth"], rotset="YZ", k=1, lr=lr)
        runs = []
        for s in range(10):
            m, info = train(ph, 1000 + s, epochs, sp); t, fl = test_block(m, n, sp)
            runs.append({"seed": 1000 + s, "val_fid_ideal": info["best_val_fid"], "test": t, "floor": fl})
        return {"lr_scores": scores, "lr": lr, "f2": Q.f2_q(ph), "runs": runs}
    if task == "oracle":
        tr = sp.states("train").to(torch.complex128); te = sp.states("test"); od = ood_states(n)
        rho_bar = torch.einsum("bi,bj->ij", tr, tr.conj()) / len(tr)
        w, v = torch.linalg.eigh(rho_bar); top = v[:, -2:].flip(1)        # dominant 2-dim subspace
        d = 2 ** n; basis = torch.zeros(d, d, dtype=torch.complex128)
        # V maps top[:,0] -> |0...0>, top[:,1] -> |1 0...0> (latent = qubit 0), completed to a unitary by QR
        M = torch.cat([top, torch.randn(d, d - 2, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))], 1)
        Qm, _ = torch.linalg.qr(M); Qm[:, :2] = top
        tgt = torch.eye(d, dtype=torch.complex128); perm = [0, d // 2] + [i for i in range(d) if i not in (0, d // 2)]
        V = tgt[:, perm] @ Qm.conj().T
        def apply(psi, pin):
            rho = Q.input_channel(Q.pure_dm(psi), pin); b = rho.shape[0]
            r = rho.reshape(b, d, d); r = V @ r @ V.conj().T
            rr = r.reshape((b,) + (2,) * (2 * n))
            for q in range(1, n):
                rr = Q.reset_qubit(rr, q, 0.0)
            r = rr.reshape(b, d, d); r = V.conj().T @ r @ V
            return Q.fidelity(psi, r.reshape((b,) + (2,) * (2 * n)))
        return {"test_ideal": {f"p_in={p}": float(apply(te, p).mean()) for p in (0.1, 0.2, 0.3)},
                "ood_ideal": float(apply(od, 0.2).mean()), "eigs_top2_share": float(w[-2:].sum().real)}
    if task == "ntrain":
        s = it["sel"]; ph = Q.QPhenotype(**s["phenotype"]); seed = s["run_seed"] * 100003 + s["eval"]
        m, info = train_noisy(ph, seed, epochs, sp); t, fl = test_block(m, n, sp)
        return {"status": "ok", "f2": s["f2"], **info, "test": t, "floor": fl}


if __name__ == "__main__":
    task, outdir, sf, sa, sv = sys.argv[1:6]; dry = "--dry" in sys.argv
    torch.set_num_threads(1); sim.set_precision("c64"); os.makedirs(outdir, exist_ok=True)
    si, sn = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
    SF, SA = json.load(open(sf)), json.load(open(sa)); SV.update(json.load(open(sv)))
    OUT = os.path.join(outdir, f"{task}_shard{si}.jsonl")
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    its = items(task, SF, SA)
    if dry:
        its = its[:: max(1, len(its) // 3)][:3]
    for j, it in enumerate(its):
        if j % sn != si or it["key"] in done:
            continue
        t0 = time.time(); res = run(task, it, 2 if dry else 100, None)
        open(OUT, "a").write(json.dumps({"key": it["key"], "n": it["n"], **res, "seconds": time.time() - t0, "dry": dry}) + "\n")
        print(task, it["key"], res.get("status", "ok"), round(time.time() - t0, 1), flush=True)  # no test numbers in the log
