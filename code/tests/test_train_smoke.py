# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
import math
import numpy as np
import torch
from qmiqcae.train import TrainCfg, lr_at, train_model, evaluate
from qmiqcae.models import build_model
from qmiqcae.circuits import CircuitSpec
from qmiqcae import sim

def test_lr_schedule_endpoints():
    cfg = TrainCfg(lr_max=1e-2, warmup_frac=0.05, lr_min_frac=0.05)
    T = 1000
    assert abs(lr_at(0, T, cfg) - 1e-2 / 50) < 1e-12
    assert abs(lr_at(49, T, cfg) - 1e-2) < 1e-12
    assert abs(lr_at(T - 1, T, cfg) - 5e-4) < 2e-6
    assert max(lr_at(s, T, cfg) for s in range(T)) <= 1e-2 + 1e-15

def test_training_reduces_loss_and_restores_best():
    torch.manual_seed(0)
    x = torch.zeros(256, 1, 28, 28); x[:, :, 8:20, 8:20] = 1.0
    x = x * torch.rand(256, 1, 1, 1)
    vn = (x[:64] + 0.3 * torch.randn(64, 1, 28, 28)).clamp(0, 1); vc = x[:64]
    m = build_model("qcae", 4, 8, CircuitSpec(4, 1, "ring"), seed=0)
    cfg = TrainCfg(lr_max=1e-2, max_epochs=6, patience=10, batch=32)
    m, info = train_model(m, x, vn, vc, cfg)
    h = info["history"]
    assert h[-1]["train_mse"] < h[0]["train_mse"]
    ev = evaluate(m, vn, vc)
    assert abs(ev["ssim"] - info["best_val_ssim"]) < 1e-9
    evn = evaluate(m, vn, vc, noise=sim.NoiseModel())
    assert evn["ssim"] != ev["ssim"]
