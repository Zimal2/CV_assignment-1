"""hybrid.py - Spatial + Fourier hybrid methods (part c)."""
import cv2
import numpy as np
from common import pct_stretch
import fourier as fo
import spatial as sp


def hybrid_sequential(x, homo=None, pre_sigma=0.0, stretch=(1, 99.5), gamma=1.0, clip=1.5, tiles=6,
                      sigma=1.5, k=0.5, alpha=0.8):
    """Hybrid 1 (sequential):
    [optional light Gaussian pre-smoothing] -> homomorphic filter (Fourier: global illumination/contrast)
    -> percentile stretch -> [gamma] -> CLAHE -> bilateral -> mild unsharp (spatial: local detail)
    -> blend with the (stretched) original:  I = alpha*processed + (1-alpha)*original."""
    homo = homo or dict(d0_frac=0.04, gl=0.5, gh=1.5, eps=0.1)
    xin = cv2.GaussianBlur(x, (0, 0), pre_sigma) if pre_sigma > 0 else x
    y = fo.homomorphic(xin, exp_out=False, **homo)  # log-domain variant, then spatial tone mapping
    y = pct_stretch(y, *stretch)
    y = sp.gamma_correct(y, gamma)
    y = sp.clahe(y, clip, tiles)
    y = sp.denoise(y, "bilateral")
    y = sp.unsharp(y, sigma, k)
    orig = pct_stretch(x, 0.5, 99.5)
    return np.clip(alpha * y + (1 - alpha) * orig, 0, 1).astype(np.float32)


def hybrid_fusion(I_spatial, I_fourier, w=0.6):
    """Hybrid 2 (weighted fusion): I = w*I_spatial + (1-w)*I_fourier."""
    return np.clip(w * I_spatial + (1 - w) * I_fourier, 0, 1).astype(np.float32)
