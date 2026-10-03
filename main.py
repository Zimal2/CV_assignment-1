"""main.py - runs parts (a)-(d) for Chest and Skeleton; saves figures, CSV metrics and runtimes.
Usage: python main.py <input_dir> <output_dir>"""
import os, sys, json
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_image, to_u8, compute_metrics, median_time_ms
import spatial as sp, fourier as fo, hybrid as hy

IN = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)

# Per-image tuned parameters (chosen after the sweeps below + visual inspection)
CFG = {
 "Chest": dict(
    sp=dict(lo=1, hi=99.5, gamma=0.8, clip=1.5, tiles=6, sigma=1.5, k=0.7, pre_sigma=1.0),
    hfe=dict(d0_frac=0.05, a=0.8, b=1.5, kind="butter", n=2, norm=None),
    hp=dict(d0_frac=0.05),
    homo=dict(d0_frac=0.04, gl=0.4, gh=1.8, eps=0.1),
    hy=dict(homo=dict(d0_frac=0.04, gl=0.6, gh=1.4, eps=0.1), pre_sigma=1.8, stretch=(1, 99.5),
            gamma=0.9, clip=1.2, tiles=4, sigma=2.0, k=0.5, alpha=0.8), w=0.6),
 "Skeleton": dict(
    sp=dict(lo=0.5, hi=99.5, gamma=0.6, clip=1.5, tiles=6, sigma=1.5, k=0.7, pre_sigma=0.0),
    hfe=dict(d0_frac=0.05, a=0.8, b=1.5, kind="butter", n=2, norm=None),
    hp=dict(d0_frac=0.05),
    homo=dict(d0_frac=0.04, gl=0.4, gh=1.8, eps=0.1),
    hy=dict(homo=dict(d0_frac=0.04, gl=0.6, gh=1.4, eps=0.3), pre_sigma=0.0, stretch=(1, 99.5),
            gamma=0.7, clip=1.2, tiles=4, sigma=2.0, k=0.4, alpha=0.8), w=0.6),
}

def save(name, im):
    cv2.imwrite(os.path.join(OUT, name + ".png"), to_u8(im))

def grid(images, titles, path, ncols=4, figsize=None):
    n = len(images); nr = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nr, ncols, figsize=figsize or (4 * ncols, 4 * nr))
    axs = np.atleast_1d(axs).ravel()
    for a in axs: a.axis("off")
    for a, im, t in zip(axs, images, titles):
        a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

all_rows, sweep_rows, runs, fusion_only = [], [], {}, {}
for name in ["Chest", "Skeleton"]:
    for ext in (".bmp", ".jpg", ".png"):          # prefer the assignment's BMP inputs
        path = os.path.join(IN, name + ext)
        if os.path.exists(path):
            break
    x = load_image(path); print(f"{name}: loaded {path}")
    c = CFG[name]; H, W = x.shape
    save(f"{name}_00_original", x)

    # ---------------- (a) spatial ----------------
    t_he, ts_he, he = median_time_ms(sp.hist_eq, x)
    t_sp, ts_sp, Is = median_time_ms(sp.spatial_enhance, x, **c["sp"])
    # step-by-step stages (for the report)
    ps = c["sp"]["pre_sigma"]
    x_pre = cv2.GaussianBlur(x, (0, 0), ps) if ps > 0 else x
    s1 = sp.pct_stretch(x_pre, c["sp"]["lo"], c["sp"]["hi"]); s2 = sp.gamma_correct(s1, c["sp"]["gamma"])
    s3 = sp.clahe(s2, c["sp"]["clip"], c["sp"]["tiles"]); s4 = sp.denoise(s3, "bilateral")
    s5 = sp.unsharp(s4, c["sp"]["sigma"], c["sp"]["k"])
    assert np.allclose(s5, Is, atol=1e-6), "stage figure must reproduce spatial_enhance()"
    grid([x, x_pre, s1, s2, s3, s4, s5],
         ["Original", "0 pre-smoothing" if ps > 0 else "0 (none)", "1 percentile stretch", "2 gamma", "3 CLAHE", "4 bilateral denoise", "5 unsharp = final"],
         f"{OUT}/{name}_a_stages.png", ncols=4, figsize=(16, 4 * 2 * (H / W) + 1))
    # spatial sweeps: gamma x clip
    sw, swt = [], []
    for g in (0.6, 0.8, 1.0, 1.2):
        for cl in (1.5, 2.0, 3.0):
            p = dict(c["sp"]); p.update(gamma=g, clip=cl)
            y = sp.spatial_enhance(x, **p); m = compute_metrics(y)
            sw.append(y); swt.append(f"g={g} clip={cl}")
            sweep_rows.append(dict(image=name, method="spatial", params=f"gamma={g}; clip={cl}", **m))
    grid(sw, swt, f"{OUT}/{name}_a_sweep_gamma_clip.png", ncols=4, figsize=(16, 4 * 3 * (H / W) + 1))

    # ---------------- (b) Fourier ----------------
    t_hp, ts_hp, Ihp = median_time_ms(fo.highpass_baseline, x, **c["hp"])
    t_hfe, ts_hfe, If = median_time_ms(fo.hfe, x, **c["hfe"])
    t_ho, ts_ho, Iho = median_time_ms(fo.homomorphic, x, **c["homo"])
    # spectrum + filter mask + filtered spectrum
    xp, ph, pw = fo._pad(x)
    d0 = fo.d0_from_frac(x.shape, c["hfe"]["d0_frac"]) * 1.2
    Hmask = c["hfe"]["a"] + c["hfe"]["b"] * fo.butterworth_hp(xp.shape, d0, c["hfe"]["n"])
    S0 = fo.spectrum(xp)
    S1 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(xp)) * Hmask))
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    for a, im, t in zip(axs, [x, S0, Hmask, S1], ["Original", "Log-magnitude spectrum", "Butterworth HFE mask H = a + b*H_hp", "Filtered spectrum"]):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_b_spectrum_filter.png", dpi=110); plt.close()
    # Fourier sweeps: D0 x kind
    sw, swt = [], []
    for kind, n in [("gaussian", 2), ("butter", 1), ("butter", 2), ("butter", 4)]:
        for df in (0.02, 0.05, 0.10):
            p = dict(c["hfe"]); p.update(d0_frac=df, kind=kind, n=n)
            y = fo.hfe(x, **p); m = compute_metrics(y)
            lab = f"{kind}{'' if kind=='gaussian' else ' n='+str(n)} D0={df}"
            sw.append(y); swt.append(lab)
            sweep_rows.append(dict(image=name, method="fourier_hfe", params=lab, **m))
    grid(sw, swt, f"{OUT}/{name}_b_sweep_d0_order.png", ncols=3, figsize=(12, 4 * 4 * (H / W) + 1))

    # ---------------- (c) hybrid ----------------
    t_h1, ts_h1, Ih1 = median_time_ms(hy.hybrid_sequential, x, **c["hy"])
    sw, swt = [], []
    for al in (0.7, 0.8, 0.9):
        p = dict(c["hy"]); p["alpha"] = al
        y = hy.hybrid_sequential(x, **p); sw.append(y); swt.append(f"Hybrid1 alpha={al}")
        sweep_rows.append(dict(image=name, method="hybrid1", params=f"alpha={al}", **compute_metrics(y)))
    for w in (0.5, 0.6, 0.7):
        y = hy.hybrid_fusion(Is, If, w); sw.append(y); swt.append(f"Hybrid2 w={w}")
        sweep_rows.append(dict(image=name, method="hybrid2", params=f"w={w}", **compute_metrics(y)))
    grid(sw, swt, f"{OUT}/{name}_c_sweep_hybrids.png", ncols=3, figsize=(12, 4 * 2 * (H / W) + 1))

    def fuse(): return hy.hybrid_fusion(Is, If, c["w"])
    t_h2, ts_h2, Ih2 = median_time_ms(fuse)
    t_h2_fusion_only = t_h2
    # END-TO-END cost (includes producing both source images). Taken as the median of the per-run totals
    # (spatial_i + HFE_i + fusion_i) rather than the sum of the three medians, so that the figure quoted in
    # the report equals the median of the 15 logged end-to-end runs. Fusion-only cost kept separately.
    ts_h2 = [a + b + c_ for a, b, c_ in zip(ts_h2, ts_sp, ts_hfe)]
    t_h2 = float(np.median(ts_h2))

    runs[name] = {"Histogram equalization": ts_he, "Spatial (final)": ts_sp, "Gaussian high-pass": ts_hp,
                  "Fourier HFE (final)": ts_hfe, "Homomorphic": ts_ho, "Hybrid 1 (sequential)": ts_h1,
                  "Hybrid 2 (fusion)": ts_h2}
    fusion_only[name] = t_h2_fusion_only
    results = {"Original": (x, None), "Histogram equalization": (he, t_he), "Spatial (final)": (Is, t_sp),
               "Gaussian high-pass": (Ihp, t_hp), "Fourier HFE (final)": (If, t_hfe), "Homomorphic": (Iho, t_ho),
               "Hybrid 1 (sequential)": (Ih1, t_h1), "Hybrid 2 (fusion)": (Ih2, t_h2)}
    for k, (im, t) in results.items():
        if k != "Original":          # already written above as {name}_00_original.png
            save(f"{name}_{k.replace(' ', '_').replace('(', '').replace(')', '')}", im)
        m = compute_metrics(im)
        all_rows.append(dict(image=name, method=k, runtime_ms=t, **m))

    # main comparison figure + histograms + zoom crops
    keys = list(results)
    grid([results[k][0] for k in keys], keys, f"{OUT}/{name}_d_comparison.png", ncols=4, figsize=(16, 4 * 2 * (H / W) + 1))
    fig, axs = plt.subplots(2, 4, figsize=(18, 7))
    for a, k in zip(axs.ravel(), keys):
        a.hist(to_u8(results[k][0]).ravel(), bins=64, range=(0, 255), color="k", log=True)
        a.set_title(k, fontsize=10); a.set_xlabel("intensity"); a.set_ylabel("count (log)")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_d_histograms.png", dpi=110); plt.close()
    # zoom crop: central region
    cy, cx = int(H * 0.30), int(W * 0.30); ch, cw = int(H * 0.35), int(W * 0.40)
    crops = [results[k][0][cy:cy + ch, cx:cx + cw] for k in keys]
    grid(crops, keys, f"{OUT}/{name}_d_zoom_crops.png", ncols=4, figsize=(16, 4 * 2 * (ch / cw) + 1))

df = pd.DataFrame(all_rows); df.to_csv(f"{OUT}/metrics.csv", index=False)
pd.DataFrame(sweep_rows).to_csv(f"{OUT}/sweeps.csv", index=False)
runs["fusion_only_ms"] = fusion_only
json.dump(runs, open(f"{OUT}/runtimes_ms.json", "w"), indent=1)
json.dump(CFG, open(f"{OUT}/final_params.json", "w"), indent=1)
pd.set_option("display.width", 200); print(df.round(4).to_string(index=False))
