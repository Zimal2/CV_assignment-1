"""fourier.py - Fourier-domain-only enhancement (part b). All processing is done on the spectrum."""
import numpy as np
from common import pct_stretch


def _pad(x, frac=0.1):
    """Reflect-pad to reduce wrap-around (boundary) artefacts of the DFT."""
    ph, pw = int(x.shape[0] * frac), int(x.shape[1] * frac)
    return np.pad(x, ((ph, ph), (pw, pw)), mode="reflect"), ph, pw


def dist_grid(shape):
    """Distance D(u,v) from the centre of the shifted spectrum."""
    r, c = shape
    u, v = np.meshgrid(np.arange(c) - c // 2, np.arange(r) - r // 2)
    return np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)


def spectrum(x):
    """Log-magnitude spectrum for display."""
    F = np.fft.fftshift(np.fft.fft2(x))
    return np.log1p(np.abs(F))


def gaussian_hp(shape, d0):
    D = dist_grid(shape)
    return 1.0 - np.exp(-(D ** 2) / (2.0 * d0 ** 2))


def butterworth_hp(shape, d0, n=2):
    D = dist_grid(shape)
    return 1.0 / (1.0 + (d0 / (D + 1e-6)) ** (2 * n))


def minmax(x):
    """Visualisation scaling only (affine min-max to [0,1]); NOT an enhancement step."""
    a, b = float(x.min()), float(x.max())
    return ((x - a) / (b - a + 1e-12)).astype(np.float32)


def signed_display(x):
    """Zero-centred display of a signed high-pass result: 0.5 + x / (2 max|x|)."""
    return (0.5 + x / (2.0 * np.abs(x).max() + 1e-12)).astype(np.float32)


def apply_filter(x, make_H, norm=None, signed=False):
    """Pad -> FFT -> shift -> multiply -> inverse -> crop -> display scaling.
    norm=None -> pure Fourier-only output with min-max visualisation scaling.
    norm=(lo,hi) -> percentile stretch (a spatial point operation; only used inside hybrids)."""
    xp, ph, pw = _pad(x)
    H = make_H(xp.shape)
    F = np.fft.fftshift(np.fft.fft2(xp))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    out = out[ph:ph + x.shape[0], pw:pw + x.shape[1]]
    if signed:
        return signed_display(out)
    return minmax(out) if norm is None else pct_stretch(out, *norm)


def d0_from_frac(shape, frac):
    return frac * min(shape)


def highpass_baseline(x, d0_frac=0.05):
    """Plain Gaussian high-pass (negative-result baseline: removes DC/low frequencies)."""
    d0 = d0_from_frac(x.shape, d0_frac)
    return apply_filter(x, lambda s: gaussian_hp(s, d0 * (1 + 0.2)), signed=True)


def hfe(x, d0_frac=0.05, a=0.7, b=1.5, kind="gaussian", n=2, norm=None):
    """High-frequency emphasis H = a + b*H_hp (Gaussian or Butterworth)."""
    d0 = d0_from_frac(x.shape, d0_frac) * 1.2  # *1.2 compensates for padding

    def make(s):
        Hhp = gaussian_hp(s, d0) if kind == "gaussian" else butterworth_hp(s, d0, n)
        return a + b * Hhp
    return apply_filter(x, make, norm)


def homomorphic(x, d0_frac=0.05, gl=0.5, gh=1.8, c=1.0, eps=0.03, norm=None, exp_out=True):
    """Homomorphic filter: log -> FFT -> H -> IFFT -> exp (corrects illumination, boosts reflectance)."""
    d0 = d0_from_frac(x.shape, d0_frac) * 1.2
    xp, ph, pw = _pad(x)
    L = np.log(xp.astype(np.float64) + eps)  # eps avoids log(0) and sets how strongly dark pixels are expanded
    D = dist_grid(xp.shape)
    H = (gh - gl) * (1.0 - np.exp(-c * D ** 2 / d0 ** 2)) + gl
    F = np.fft.fftshift(np.fft.fft2(L))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    out = out[ph:ph + x.shape[0], pw:pw + x.shape[1]]
    # exp_out=True: classical homomorphic filter, g = exp(IDFT{H * DFT{log(f+eps)}}) - eps, min-max display scaling.
    # exp_out=False: "modified log-domain homomorphic enhancement" - the result stays in the log domain
    # (monotonic, so only the tone curve differs) because bright hot spots make the exp output heavy-tailed.
    if exp_out:
        out = np.exp(out) - eps
    out = out.astype(np.float32)
    return minmax(out) if norm is None else pct_stretch(out, *norm)
