"""common.py - shared utilities: loading, normalisation, metrics, timing, plotting.
References: Gonzalez & Woods, Digital Image Processing (4th ed.); OpenCV documentation."""
import time
import cv2
import numpy as np
from skimage.measure import shannon_entropy


def load_image(path):
    """Load as grayscale float32 in [0,1] (JPEG/BMP both supported)."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    g = strip_white_border(g)
    return g.astype(np.float32) / 255.0


def strip_white_border(g, thr=245, max_px=8):
    """Remove a saturated white frame (rows/cols whose mean >= thr) from the image edges.
    The supplied images carry a 1-4 px white frame that otherwise causes edge ringing in the
    Fourier methods and skews percentile statistics. Only edge rows/columns are removed."""
    t, b, l, r = 0, g.shape[0], 0, g.shape[1]
    while t < max_px and g[t, l:r].mean() >= thr: t += 1
    while max_px > g.shape[0] - b and g[b - 1, l:r].mean() >= thr: b -= 1
    while l < max_px and g[t:b, l].mean() >= thr: l += 1
    while max_px > g.shape[1] - r and g[t:b, r - 1].mean() >= thr: r -= 1
    return g[t:b, l:r]


def to_u8(x):
    """Clip a [0,1] float image and convert to uint8."""
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def pct_stretch(x, lo=1, hi=99):
    """Robust contrast stretching using percentiles instead of min/max."""
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-6:
        return x.copy()
    return np.clip((x - a) / (b - a), 0, 1).astype(np.float32)


def compute_metrics(x):
    """Return dict of objective measures for a float [0,1] image.
    - mean, std (global contrast), entropy (bits), avg_gradient, tenengrad (mean Sobel magnitude),
    - noise_sigma (Immerkaer fast noise estimate), edge_density (Canny pixels / total)."""
    u8 = to_u8(x)
    xf = x.astype(np.float64)
    gy, gx = np.gradient(xf)
    avg_grad = np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2))
    sx = cv2.Sobel(xf, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(xf, cv2.CV_64F, 0, 1, ksize=3)
    teneng = np.mean(np.hypot(sx, sy))
    # Immerkaer (1996) noise estimator
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = xf.shape
    conv = cv2.filter2D(xf, -1, k)[1:-1, 1:-1]
    sigma = np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2))
    edges = cv2.Canny(u8, 50, 150)
    return dict(mean=float(xf.mean()), std=float(xf.std()),
                entropy=float(shannon_entropy(u8)), avg_gradient=float(avg_grad),
                tenengrad=float(teneng), noise_sigma=float(sigma),
                edge_density=float((edges > 0).mean()))


def median_time_ms(fn, *args, runs=15, **kw):
    """Median wall-clock time (ms) of fn(*args) over several runs; also returns last result."""
    ts, out = [], None
    fn(*args, **kw)  # warm-up (excluded)
    for _ in range(runs):
        t = time.perf_counter()
        out = fn(*args, **kw)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), ts, out
