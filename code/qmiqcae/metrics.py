# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Image metrics (F7.1-F7.2). SSIM matches skimage.metrics.structural_similarity with
gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1 (reflect padding, border crop)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

K1, K2, SIGMA_W, TRUNCATE = 0.01, 0.03, 1.5, 3.5


def _gauss_kernel(sigma: float = SIGMA_W, truncate: float = TRUNCATE) -> torch.Tensor:
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float64)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return g


def ssim_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """x, y: (B, 1, H, W) in [0, data_range]. Returns (B,) SSIM, mean over the cropped SSIM map."""
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    g = _gauss_kernel()
    r = (g.numel() - 1) // 2
    kh = g.view(1, 1, 1, -1)
    kv = g.view(1, 1, -1, 1)

    def filt(t):
        t = F.pad(t, (r, r, 0, 0), mode="reflect")
        t = F.conv2d(t, kh)
        t = F.pad(t, (0, 0, r, r), mode="reflect")
        return F.conv2d(t, kv)

    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2
    ux, uy = filt(x), filt(y)
    uxx, uyy, uxy = filt(x * x), filt(y * y), filt(x * y)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    s = s[..., r:-r, r:-r]
    return s.mean(dim=(1, 2, 3))


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = ((x.to(torch.float64) - y.to(torch.float64)) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range ** 2 / mse)
