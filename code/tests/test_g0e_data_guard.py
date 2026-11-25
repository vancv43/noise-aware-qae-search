# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""G0-e: split disjointness, sizes, and the test split is never opened in search/pilot mode."""
import numpy as np
import pytest
import torch
from qmiqcae.data import make_splits, SplitStore, TestAccessError
from qmiqcae.experiment import make_search_evaluator
from qmiqcae.genome import decode, D
from qmiqcae.problem import Problem

def synth(n, seed):
    r = np.random.default_rng(seed)
    return (r.random((n, 28, 28)) * 255).astype(np.uint8), r.integers(0, 10, n)

def small_splits():
    trx, try_ = synth(3000, 0); tex, tey = synth(1000, 1)
    return make_splits(trx, try_, tex, tey, seed=0, sizes={"train": 200, "val": 50, "test": 100})

def test_splits_disjoint_and_sized():
    s = small_splits()
    m = s["_meta"]
    assert len(np.intersect1d(m["idx_train"], m["idx_val"])) == 0
    assert len(s["train"][0]) == 200 and len(s["val"][0]) == 50 and len(s["test"][0]) == 100
    assert m["train_test_pixel_duplicates"] == 0

def test_test_access_blocked_in_search_and_pilot():
    s = small_splits()
    for mode in ("search", "pilot"):
        st = SplitStore(s, mode=mode)
        with pytest.raises(TestAccessError):
            st.get("test")
    assert SplitStore(s, mode="final").get("test")[0].shape[0] == 100

def test_search_evaluator_never_touches_test():
    s = small_splits()
    st = SplitStore(s, mode="search")
    fn = make_search_evaluator(st, None, epochs=1, sigma=0.5, run_seed=0)
    prob = Problem(fn, budget=2)
    rng = np.random.default_rng(0)
    g = rng.random(D); g[2] = 0.0  # n = 4 qubits for speed
    prob(g)
    assert "test" not in st.accessed and set(st.accessed) <= {"train", "val"}
    assert prob.records[0]["status"] == "ok"
