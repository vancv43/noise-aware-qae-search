# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Training core (F2.1-F2.6) and evaluation (F7.*).

Mini-batch AdamW, linear warmup then cosine decay to lr_min_frac*lr_max (per step), early stopping on validation
SSIM (ideal simulation) with best-weight restore, per-epoch train/val logging and fit flags.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

from . import sim
from .data import add_noise
from .metrics import ssim_per_image, psnr_per_image
from .models import set_noise


@dataclass
class TrainCfg:
    lr_max: float = 3e-3
    weight_decay: float = 1e-4
    batch: int = 64
    max_epochs: int = 100
    warmup_frac: float = 0.05
    lr_min_frac: float = 0.05
    patience: int = 15
    sigma: float = 0.5
    seed: int = 0
    eval_batch: int = 250
    val_noise_seed: int = 1001


def lr_at(step: int, total: int, cfg: TrainCfg) -> float:
    """F2.2 warmup + cosine."""
    w = max(1, int(round(cfg.warmup_frac * total)))
    if step < w:
        return cfg.lr_max * (step + 1) / w
    t = (step - w) / max(1, total - w)
    lo = cfg.lr_min_frac * cfg.lr_max
    return lo + 0.5 * (cfg.lr_max - lo) * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model: nn.Module, x_noisy: torch.Tensor, x_clean: torch.Tensor, noise: Optional[sim.NoiseModel] = None,
             shots: Optional[int] = None, shot_seed: Optional[int] = None, batch: int = 250):
    model.eval()
    set_noise(model, noise, shots, shot_seed)
    s, p = [], []
    try:
        for i in range(0, len(x_noisy), batch):
            out = model(x_noisy[i:i + batch])
            s.append(ssim_per_image(out, x_clean[i:i + batch]))
            p.append(psnr_per_image(out, x_clean[i:i + batch]))
    finally:
        set_noise(model, None)
    s = torch.cat(s)
    p = torch.cat(p)
    return {"ssim": float(s.mean()), "psnr": float(p.mean()), "ssim_per_image": s}


def train_model(model: nn.Module, x_train: torch.Tensor, val_noisy: torch.Tensor, val_clean: torch.Tensor,
                cfg: TrainCfg, max_epochs: Optional[int] = None, log_every: int = 0):
    """Returns (model with best-val weights, info dict). Uses ideal simulation for training and validation."""
    torch.manual_seed(cfg.seed)
    gen_noise = torch.Generator().manual_seed(cfg.seed * 7919 + 17)
    gen_perm = torch.Generator().manual_seed(cfg.seed * 104729 + 3)
    epochs = max_epochs or cfg.max_epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr_max, weight_decay=cfg.weight_decay)  # F2.1 decoupled wd
    n = len(x_train)
    steps_per_epoch = math.ceil(n / cfg.batch)
    total = steps_per_epoch * epochs
    step = 0
    best, best_state, best_epoch, bad = -1e9, None, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen_perm)
        tot, cnt = 0.0, 0
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            if len(idx) < 2:
                continue  # BatchNorm needs >= 2
            xc = x_train[idx]
            xn = add_noise(xc, cfg.sigma, gen_noise)  # F2.4 fresh noise each epoch
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            out = model(xn)
            loss = ((out - xc) ** 2).mean()  # F2.3 MSE
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            tot += loss.item() * len(idx)
            cnt += len(idx)
        ev = evaluate(model, val_noisy, val_clean, batch=cfg.eval_batch)
        hist.append({"epoch": ep + 1, "train_mse": tot / cnt, "val_ssim": ev["ssim"], "val_psnr": ev["psnr"],
                     "lr_end": opt.param_groups[0]["lr"]})
        if log_every and (ep + 1) % log_every == 0:
            print(hist[-1], flush=True)
        if ev["ssim"] > best + 1e-6:
            best, best_epoch, bad = ev["ssim"], ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    info = {"best_val_ssim": best, "best_epoch": best_epoch, "epochs_run": len(hist),
            "history": hist, "seconds": time.time() - t0, "cfg": asdict(cfg), "flags": fit_flags(hist, best_epoch, cfg)}
    return model, info


def fit_flags(hist: list, best_epoch: int, cfg: TrainCfg) -> dict:
    """F2.6 underfit / overfit / early-peak flags."""
    tr = [h["train_mse"] for h in hist]
    va = [h["val_ssim"] for h in hist]
    underfit = len(tr) > 1 and tr[-1] > 0.5 * tr[0]
    overfit = False
    if best_epoch < len(hist):
        after_tr = tr[best_epoch - 1:]
        after_va = va[best_epoch - 1:]
        overfit = (after_tr[-1] < after_tr[0]) and (after_va[-1] < after_va[0] - 0.01) and \
                  (len(hist) - best_epoch) >= cfg.patience // 2
    return {"underfit": bool(underfit), "overfit": bool(overfit), "early_peak": bool(best_epoch < 10)}
