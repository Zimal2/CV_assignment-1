"""spatial.py - spatial-domain-only enhancement (part a)."""
import cv2
import numpy as np
from common import pct_stretch, to_u8


def hist_eq(x):
    """Baseline: global histogram equalisation."""
    return cv2.equalizeHist(to_u8(x)).astype(np.float32) / 255.0


def gamma_correct(x, g):
    return np.power(np.clip(x, 0, 1), g).astype(np.float32)


def clahe(x, clip=2.0, tiles=8):
    # tiles x tiles grid; smaller clip limit = less noise amplification
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return c.apply(to_u8(x)).astype(np.float32) / 255.0


def denoise(x, mode="bilateral"):
    """Light edge-preserving denoising (bilateral or median)."""
    u8 = to_u8(x)
    if mode == "median":
        y = cv2.medianBlur(u8, 3)
    else:
        y = cv2.bilateralFilter(u8, d=5, sigmaColor=25, sigmaSpace=3)
    return y.astype(np.float32) / 255.0


def unsharp(x, sigma=1.5, k=0.8):
    """Unsharp masking: I + k (I - G_sigma * I)."""
    blur = cv2.GaussianBlur(x, (0, 0), sigma)
    return np.clip(x + k * (x - blur), 0, 1).astype(np.float32)


def spatial_enhance(x, lo=1, hi=99.5, gamma=0.8, clip=1.5, tiles=6, den="bilateral", sigma=1.5, k=0.7, pre_sigma=0.0):
    """Final spatial pipeline: [light pre-smoothing] -> percentile stretch -> gamma -> CLAHE -> denoise -> unsharp.
    pre_sigma > 0 dithers the coarse intensity quantisation of the source images (Chest.bmp carries a
    219-entry grey palette, and only ~20 distinct levels fall in the darkest 40% of the image; Skeleton.bmp
    has ~9). Without it CLAHE stretches those few levels into wide flat plateaus, i.e. visible posterisation
    (contour banding) in the dark lung/background regions."""
    if pre_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), pre_sigma)
    y = pct_stretch(x, lo, hi)
    y = gamma_correct(y, gamma)
    y = clahe(y, clip, tiles)
    y = denoise(y, den)
    return unsharp(y, sigma, k)
