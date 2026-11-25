# Option B pilot: pre-specified pass criterion (written before the pilot is started)
Context: the TFIM family h in [0.2, 2] is solved better by an input-independent fixed state than by any QAE
(audit_rev2.py, rev2 kernels). Numeric screening (taskB_screen.py, ideal gates) kept two families:
  F3 mixed-field Ising H = -sum ZZ - h sum X - g sum Z, h ~ U[0.2,1.2], g ~ U[-0.3,0.3]
  F5 mixture: TFIM h ~ U[0.2,2] or XXZ (sum XX+YY+D ZZ) D ~ U[-0.5,2], with probability 1/2 each
Pilot (validation states only, main hardware noise, p_in = 0.2, ideal training, 100 epochs, seed 0):
  QAE grid: topology {ring, linear}, depth {1,2,3}, rotset {YZ, XYZ}, k {1,2}, lr {0.03, 0.1}  (48 per family and n)
  fixed-output control: topology {ring, linear, all}, depth {1,2,3}, XYZ, lr {0.03, 0.1}    (18 per family and n)
PASS for a family if, for both n = 4 and n = 6:
  best QAE noisy validation fidelity >= best fixed-output noisy validation fidelity + 0.03, and
  best QAE noisy validation fidelity >= do-nothing floor + 0.05.
If both pass, F3 is preferred (single physical model). If neither passes, fall back to option A.
The test split is not generated or read in the pilot.
