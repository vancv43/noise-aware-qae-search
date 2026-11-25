# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Datasets, disjoint splits, noise injection and the test-access guard (F2.4, G0-e).

Splits (the study design): train 4,000 / val 500 drawn class-stratified from the official training set; test 1,000
class-stratified from the official test set. Train noise is redrawn every epoch; val and test noise are fixed by seed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

SPLIT_SIZES = {"train": 4000, "val": 500, "test": 1000}


class TestAccessError(RuntimeError):
    pass


def _stratified_pick(labels: np.ndarray, n: int, rng: np.random.Generator, exclude: Optional[np.ndarray] = None):
    idx = np.arange(len(labels))
    if exclude is not None:
        mask = np.ones(len(labels), bool)
        mask[exclude] = False
        idx = idx[mask]
    classes = np.unique(labels[idx])
    per = n // len(classes)
    rem = n - per * len(classes)
    out = []
    for i, c in enumerate(classes):
        ci = idx[labels[idx] == c]
        k = per + (1 if i < rem else 0)
        if k > len(ci):
            raise ValueError(f"class {c}: need {k}, have {len(ci)}")
        out.append(rng.choice(ci, size=k, replace=False))
    return np.sort(np.concatenate(out))


def make_splits(train_x, train_y, test_x, test_y, seed: int = 0, sizes=SPLIT_SIZES):
    """Return dict split -> (x float32 (N,1,28,28) in [0,1], y int64). Disjointness asserted."""
    rng = np.random.default_rng(seed)
    tr = _stratified_pick(train_y, sizes["train"], rng)
    va = _stratified_pick(train_y, sizes["val"], rng, exclude=tr)
    te = _stratified_pick(test_y, sizes["test"], rng)
    assert len(np.intersect1d(tr, va)) == 0
    out = {
        "train": (train_x[tr], train_y[tr]),
        "val": (train_x[va], train_y[va]),
        "test": (test_x[te], test_y[te]),
    }
    # content-level duplicate check between train/val and test (hash of pixels)
    h = lambda a: {hashlib.md5(r.tobytes()).hexdigest() for r in a}
    htest = h(out["test"][0])
    dup = len(h(out["train"][0]) & htest) + len(h(out["val"][0]) & htest)
    out["_meta"] = {"idx_train": tr, "idx_val": va, "idx_test": te, "train_test_pixel_duplicates": dup}
    for k in ("train", "val", "test"):
        assert len(out[k][0]) == sizes[k]
    return out


def to_tensor(x: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x.float()


def add_noise(x: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """F2.4 x~ = clip(x + sigma*eps, 0, 1)."""
    return (x + sigma * torch.randn(x.shape, generator=gen)).clamp(0, 1)


@dataclass
class SplitStore:
    """Holds the three splits. In mode 'search' or 'pilot' any test access raises TestAccessError."""
    splits: dict
    mode: str = "search"
    accessed: list = field(default_factory=list)

    def get(self, name: str):
        if name == "test" and self.mode in ("search", "pilot"):
            raise TestAccessError(f"test split requested in mode={self.mode}")
        self.accessed.append(name)
        x, y = self.splits[name]
        return to_tensor(x), torch.as_tensor(y)

    def fixed_noisy(self, name: str, sigma: float, seed: int):
        x, y = self.get(name)
        g = torch.Generator().manual_seed(seed)
        return add_noise(x, sigma, g), x, y


def load_torchvision(name: str, root: str):
    import torchvision
    cls = {"mnist": torchvision.datasets.MNIST, "fashion": torchvision.datasets.FashionMNIST}[name]
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()


def load_organamnist(npz_path: str):
    """MedMNIST v2 OrganAMNIST 28x28 (Yang et al., Sci Data 2023). Official train+val pooled as 'training set'."""
    d = np.load(npz_path)
    trx = np.concatenate([d["train_images"], d["val_images"]])
    tr_y = np.concatenate([d["train_labels"], d["val_labels"]]).ravel()
    return trx, tr_y, d["test_images"], d["test_labels"].ravel()


def binary_subset(x, y, classes=(0, 1)):
    m = np.isin(y, classes)
    return x[m], y[m]
