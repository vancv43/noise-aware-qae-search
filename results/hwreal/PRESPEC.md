# Hardware-realism re-scoring: pre-specification (written before the run is started)
Nothing in results/hwreal exists except check_hwreal.json (code checks, no training).
Motivation: an internal review . The manuscript states that the omitted idle relaxation
and reset duration "would likely widen rather than close the gap between the noise-aware and the ideal-gate selections".
This run tests that sentence.

## Models (code/experiments/hwreal.py, code hash in the run log)
- Swarm groups of roadmap_v2: NA-QAS-QAE cheap / ideal / maxfid, MOPSO-ideal cheap / ideal; every run, n = 4, 6;
  78 distinct trained models (deduplicated), retrained with their original seeds; repro check |dval| <= 1e-5 (the reproduction tolerance),
  a model failing it is not scored.
- Input-independent control: linear, depth 2, lr 0.1 (the rev2 choice), seeds 1000-1009, n = 4, 6.

## Noise conditions (scored on the 128 new states of roadmap_v2; the test states only under the first three)
- main: unchanged main model (must reproduce the rev2 numbers).
- main+idle: + thermal relaxation of every idle qubit for the idle time of each ASAP moment
  (moment length = slowest gate: 50 ns rotation, 500 ns CNOT).
- main+idle+reset1us (PRIMARY): + reset of 1 us, during which the latent qubits relax.
- main+idle+reset4us, sher+idle+reset1us: secondary.

## Primary test (per n, new states, two-sided exact Wilcoxon signed-rank over the 10 runs, Holm over 2 per n)
For run r: D_r(c) = F[NA maxfid](c) - F[MI rule](c), rule in {cheap, ideal}; statistic dD_r = D_r(primary) - D_r(main).
Pre-declared action on the manuscript sentence:
- dD > 0 with Holm p < 0.05 at both n: keep "widen".
- dD < 0 with Holm p < 0.05 at either n: rewrite to say that the omissions narrowed the gap, with the numbers.
- otherwise: rewrite to say that adding them changed the gap by at most max|mean dD|, without significance.

## Secondary (descriptive, reported whatever they show)
- Mean fidelity of every group and the control under all conditions; floor.
- Whether the control still beats every QAE group under each condition.
- Spearman rho between each model's drop main -> primary and its CNOT count, per n.
- Repro: group means under the unchanged main model on the new states equal rev2_numbers.json (3 decimals) within 0.001.

## Known before the run (code checks, random untrained circuits, n = 4)
Extra loss main -> primary between -0.008 and +0.012, not increasing with CNOTs (rho -0.36): amplitude damping
toward |0> can raise fidelity. So the direction of the primary statistic is genuinely open.
