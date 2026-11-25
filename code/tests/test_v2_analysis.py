# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Audit tests for the v2 analysis (HV/IGD+/statistics/selection) and the final-evaluation script."""
import json, os, subprocess, sys
import numpy as np, pytest
from scipy import stats
from qmiqcae import qae as Q, sim
from qmiqcae.analysis import hypervolume, igd_plus, select_pareto_point, friedman, wilcoxon_vs
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "analysis")); sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
pymoo = pytest.importorskip("pymoo")
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
RNG = np.random.default_rng(5)


@pytest.mark.parametrize("fm", [72, 180, 112])
def test_hv_and_igd_vs_pymoo_with_v2_normalisation(fm):
    for _ in range(20):
        F = np.column_stack([RNG.uniform(0.3, 1.0, 15), RNG.choice(np.arange(0, fm + 1, 2), 15)])
        ref = np.array([1.0, 1.0 + 1.0 / fm])
        P = np.column_stack([F[:, 0], F[:, 1] / fm])
        assert abs(hypervolume(F, f2_max=fm) - HV(ref_point=ref)(P[np.all(P < ref, 1)])) < 1e-12
        Z = np.column_stack([RNG.uniform(0.3, 1.0, 8), RNG.choice(np.arange(0, fm + 1, 2), 8)])
        assert abs(igd_plus(F, Z, f2_max=fm) - IGDPlus(np.column_stack([Z[:, 0], Z[:, 1] / fm]))(P)) < 1e-12


def test_wrong_normalisation_would_change_hv():
    F = np.array([[0.5, 30.0], [0.45, 60.0]])
    assert abs(hypervolume(F, f2_max=180) - hypervolume(F)) > 1e-3  # v1 default 112 is wrong for v2: must pass f2_max


def test_statistics_match_scipy():
    T = RNG.random((10, 8))
    assert np.allclose(friedman(T), tuple(stats.friedmanchisquare(*T.T)))
    w = wilcoxon_vs(T[:, 0], {"b": T[:, 1], "c": T[:, 2]})
    assert abs(w["b"]["p"] - stats.wilcoxon(T[:, 0], T[:, 1]).pvalue) < 1e-12
    assert w["b"]["p_holm"] >= w["b"]["p"] and 0 <= w["b"]["a12"] <= 1


def test_selection_rule():
    F = np.array([[0.3, 0], [0.2, 8], [0.25, 4], [0.1, 60], [0.4, 0]])
    assert select_pareto_point(F) == 2


def test_conditions_are_the_protocol_ones():
    from final_eval_v2 import conditions
    c = conditions()
    assert len(c) == 15
    assert c["main"][0] == sim.NoiseModel() and c["ideal"][0] is None
    assert c["p2=0.02"][0].p1 == 0.002 and c["p2=0.02"][0].p2 == 0.02
    assert c["T1=T2=50us"][0].t1 == 50e-6 == c["T1=T2=50us"][0].t2
    assert c["reset=0.05"][0].e0 == 0.05 and c["p_in=0.3"][1] == 0.3 and c["p_in=0.3"][0] == sim.NoiseModel()


def test_test_split_guard_and_analysis_never_reads_test():
    with pytest.raises(Q.TestAccessError):
        Q.QSplits(4, mode="search").states("test")
    assert len(Q.QSplits(4, mode="final").states("test")) == Q.SPLIT_N["test"]
    src = open(os.path.join(HERE, "..", "analysis", "analyze_search_v2.py")).read()
    assert "states(" not in src and "QSplits" not in src


def test_end_to_end_search_analysis_final_reproduces_model(tmp_path):
    py = sys.executable
    for arm in ("namopso", "mopso_ideal", "random"):
        for s in (0, 1):
            r = subprocess.run([py, "-W", "ignore", os.path.join(HERE, "..", "experiments", "search_v2.py"), "4", arm, str(s),
                                str(tmp_path / f"{arm}_s{s}"), "--dry"], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
    from analyze_search_v2 import analyze, objectives, load
    assert analyze(str(tmp_path), n_boot=50)["n_runs"] == 0  # smoke runs are excluded by default
    A = analyze(str(tmp_path), n_boot=50, include_smoke=True)
    res = A["per_n"]["4"]
    runs, evals = load(str(tmp_path), include_smoke=True)
    for a in res["arms"]:  # HV recomputed by hand with the v2 normalisation
        for i, s in enumerate(res["seeds"]):
            assert abs(res["hv"][a]["values"][i] - hypervolume(objectives(evals[(4, a, s)]), f2_max=72)) < 1e-15
    s0 = res["selected"]["mopso_ideal/s0"]
    assert s0["selected_on"] == "ideal"
    from final_eval_v2 import evaluate_selected
    import torch
    try:
        out = evaluate_selected(s0, 4, epochs=2, dry=False)  # asserts exact reproduction of the searched model
    finally:
        sim.set_precision("c128"); torch.set_num_threads(4)  # do not leak global precision into other tests
    assert out["repro_abs_diff"] < 1e-9 and out["status"] == "ok" and len(out["test"]) == 15 and 0 < out["test"]["main"] < 1


def test_not_reproduced_model_is_not_scored_on_test():
    import final_eval_v2 as FE
    s = {"phenotype": {"n": 4, "topology": "linear", "depth": 1, "rotset": "YZ", "k": 1, "lr": 0.03}, "run_seed": 0, "eval": 0,
         "val_fid_ideal": -1.0}
    import torch
    try:
        out = FE.evaluate_selected(s, 4, epochs=1, dry=False)
    finally:
        sim.set_precision("c128"); torch.set_num_threads(4)
    assert out["status"] == "not_reproduced" and out["test"] == {}
