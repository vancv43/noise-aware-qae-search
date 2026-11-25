# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Option B, step 1: numeric screening of candidate state families (no training, ideal gates).
For each family: spectrum of the mean training state, best constant-output fidelity (input-independent baseline),
oracle QAE fidelity for k = 1, 2 (rotate the top-2^k subspace onto the latent qubits, reset the rest, rotate back),
and the do-nothing floor, at p_in = 0.1 and 0.2. A family qualifies if oracle(k=1) - constant >= 0.10 at p_in = 0.2."""
import json, sys, os
import numpy as np
X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1., -1.]).astype(complex); Y = np.array([[0, -1j], [1j, 0]]); I = np.eye(2)
def op(o, q, n):
    m = np.array([[1.0 + 0j]])
    for j in range(n): m = np.kron(m, o if j == q else I)
    return m
def ground(Hm):
    w, v = np.linalg.eigh(Hm)
    if w[1] - w[0] < 1e-6: return None            # degenerate: skip
    return v[:, 0]
def tfim(n, h, g=0.0, J=1.0):
    return (-J * sum(op(Z, q, n) @ op(Z, (q + 1) % n, n) for q in range(n)) - h * sum(op(X, q, n) for q in range(n))
            - g * sum(op(Z, q, n) for q in range(n)))
def xxz(n, D, h=0.0):
    return (sum(op(X, q, n) @ op(X, (q + 1) % n, n) + op(Y, q, n) @ op(Y, (q + 1) % n, n) + D * op(Z, q, n) @ op(Z, (q + 1) % n, n)
                for q in range(n)) - h * sum(op(Z, q, n) for q in range(n)))
def depol_all(rho, p, n):
    d = 2 ** n
    for q in range(n):
        r = rho.reshape([2] * (2 * n)); tq = np.trace(r, axis1=q, axis2=q + n)
        mixed = np.moveaxis(np.tensordot(tq, I / 2, 0), [2 * n - 2, 2 * n - 1], [q, q + n]).reshape(d, d)
        rho = (1 - 4 * p / 3) * rho + 4 * p / 3 * mixed
    return rho
def oracle(states_tr, states_te, k, p, n):
    d = 2 ** n; rho = sum(np.outer(s, s.conj()) for s in states_tr) / len(states_tr)
    w, v = np.linalg.eigh(rho); V = v[:, ::-1]                     # columns by decreasing weight
    # latent = first k qubits (most significant), trash = rest -> basis index i*2^(n-k) for latent value i
    perm = [i * 2 ** (n - k) for i in range(2 ** k)] + [j for j in range(d) if j % 2 ** (n - k) != 0]
    W = np.zeros((d, d), complex); W[perm, :] = V.conj().T          # W maps eigvec j -> basis perm[j]
    P = np.zeros((d, d)); P[[i * 2 ** (n - k) for i in range(2 ** k)], [i * 2 ** (n - k) for i in range(2 ** k)]] = 1
    fs = []
    for s in states_te:
        r = W @ depol_all(np.outer(s, s.conj()), p, n) @ W.conj().T
        r4 = r.reshape(2 ** k, 2 ** (n - k), 2 ** k, 2 ** (n - k)); lat = np.einsum("atbt->ab", r4)
        z = np.zeros_like(r4); z[:, 0, :, 0] = lat; r = W.conj().T @ z.reshape(d, d) @ W
        fs.append(np.real(s.conj() @ r @ s))
    return float(np.mean(fs)), w[::-1]
rng = np.random.default_rng(0)
def fam(name, n):
    N = 96
    if name == "F0 TFIM h[0.2,2]": P = [(rng.uniform(0.2, 2),) for _ in range(N)]; mk = lambda a: tfim(n, a[0])
    if name == "F1 TFIM |h| in [0.2,2], both signs": P = [(rng.choice([-1, 1]) * rng.uniform(0.2, 2),) for _ in range(N)]; mk = lambda a: tfim(n, a[0])
    if name == "F2 mixed-field Ising h[0.2,2], g[-0.5,0.5]": P = [(rng.uniform(0.2, 2), rng.uniform(-0.5, 0.5)) for _ in range(N)]; mk = lambda a: tfim(n, a[0], a[1])
    if name == "F3 mixed-field Ising h[0.2,1.2], g[-0.3,0.3]": P = [(rng.uniform(0.2, 1.2), rng.uniform(-0.3, 0.3)) for _ in range(N)]; mk = lambda a: tfim(n, a[0], a[1])
    if name == "F4 XXZ D[-0.5,2], field h[-1,1]": P = [(rng.uniform(-0.5, 2), rng.uniform(-1, 1)) for _ in range(N)]; mk = lambda a: xxz(n, a[0], a[1])
    if name == "F5 TFIM h[0.2,2] + XXZ D[-0.5,2] mixture": P = [(rng.integers(2), rng.uniform(0.2, 2), rng.uniform(-0.5, 2)) for _ in range(N)]; mk = lambda a: tfim(n, a[1]) if a[0] == 0 else xxz(n, a[2])
    S = [s for s in (ground(mk(a)) for a in P) if s is not None]
    return S
out = {}
for name in ("F0 TFIM h[0.2,2]", "F1 TFIM |h| in [0.2,2], both signs", "F2 mixed-field Ising h[0.2,2], g[-0.5,0.5]",
             "F3 mixed-field Ising h[0.2,1.2], g[-0.3,0.3]", "F4 XXZ D[-0.5,2], field h[-1,1]", "F5 TFIM h[0.2,2] + XXZ D[-0.5,2] mixture"):
    for n in (4, 6):
        S = fam(name, n); tr, te = S[:64], S[64:]
        rho = sum(np.outer(s, s.conj()) for s in tr) / len(tr); w = np.linalg.eigvalsh(rho)[::-1]; phi = np.linalg.eigh(rho)[1][:, -1]
        const = float(np.mean([abs(phi.conj() @ s) ** 2 for s in te]))
        row = {"states": len(S), "top4": [round(float(x), 4) for x in w[:4]], "const": round(const, 3)}
        for p in (0.1, 0.2):
            row[f"floor p{p}"] = round(float(np.mean([np.real(s.conj() @ depol_all(np.outer(s, s.conj()), p, n) @ s) for s in te])), 3)
            for k in (1, 2):
                row[f"oracle k{k} p{p}"] = round(oracle(tr, te, k, p, n)[0], 3)
        row["margin k1 p0.2"] = round(row["oracle k1 p0.2"] - const, 3); row["margin k2 p0.2"] = round(row["oracle k2 p0.2"] - const, 3)
        out[f"{name} | n{n}"] = row; print(name, "n", n, row, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "taskB_pilot", "screen.json"), "w"), indent=1)
