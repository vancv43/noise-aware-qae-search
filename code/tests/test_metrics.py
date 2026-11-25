# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from qmiqcae.metrics import ssim_per_image, psnr_per_image

def test_ssim_psnr_match_skimage():
    rng = np.random.default_rng(3)
    a = rng.random((6, 28, 28)); b = np.clip(a + rng.normal(0, 0.2, a.shape), 0, 1)
    ours = ssim_per_image(torch.tensor(b)[:, None], torch.tensor(a)[:, None]).numpy()
    ref = [structural_similarity(a[i], b[i], data_range=1.0, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False) for i in range(6)]
    assert np.max(np.abs(ours - np.array(ref))) < 1e-4
    p = psnr_per_image(torch.tensor(b)[:, None], torch.tensor(a)[:, None]).numpy()
    pr = [peak_signal_noise_ratio(a[i], b[i], data_range=1.0) for i in range(6)]
    assert np.max(np.abs(p - np.array(pr))) < 1e-8
    assert abs(ssim_per_image(torch.tensor(a)[:, None], torch.tensor(a)[:, None]).min().item() - 1) < 1e-12
