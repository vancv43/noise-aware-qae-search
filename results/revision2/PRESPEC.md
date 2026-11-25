# Pre-specification of the confirmatory evaluation (written before the runs are started)
Nothing in results/revision2 exists yet when this file is written.

Fresh states: 128 new TFIM ground states per n, h ~ U[0.2, 2.0], numpy default_rng(seed 15), disjoint from train/val/test/OOD.
Models: the same 287 selections already defined (160 pre-declared + 127 rule alternatives), retrained with their original seeds.

Deployment rule fixed now (from validation data only, as in Appendix B):
  R1 (primary): for a search scored under noise, deploy the front point with the highest noisy validation fidelity (maxfid).
  R2 (fallback, M5): if the deployed model's noisy validation fidelity <= the validation do-nothing baseline, deploy the identity.
Primary confirmatory family (Holm over 2, two-sided exact Wilcoxon paired by run index, alpha 0.05), per n, main noise, fresh states:
  H1: NA-QAS-QAE + R1 vs MOPSO-ideal + pre-declared cheaper-half rule.
  H2: NA-QAS-QAE + R1 vs MOPSO-ideal + ideal-point rule.
Secondary (descriptive, with bootstrap CIs, no claims of significance): same at p2=0.05 (joint p1=p2/10), Sherbrooke;
  all 5 multi-objective methods under R1; success rate vs do-nothing baseline under every rule; effect of R2.
Noise isolation (M7): original test states, p2 in {0.005, 0.02, 0.05} with p1 fixed at 1e-3; p1 in {1e-4, 5e-3} with p2 fixed at 0.01.
Training-seed robustness (M10): swarm-variant selections (NA cheap/ideal/maxfid, MOPSO-ideal cheap/ideal), 3 fresh seeds 7001-7003.
Fixed-output control (M4): zero latent qubits (all qubits reset, decoder prepares one trained state), structure/lr chosen on noisy validation, 10 seeds.
