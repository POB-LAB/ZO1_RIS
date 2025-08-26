import numpy as np
import cv2


def _rolling_mean(x, w):
    if w <= 1:
        return x.astype(np.float32)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), k, mode='same')


def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-6


def _local_prominence(signal, idx, win):
    n = len(signal)
    left = signal[max(0, idx - win):idx + 1]
    right = signal[idx:min(n, idx + win + 1)]
    if left.size == 0 or right.size == 0:
        return 0.0
    left_min = left.min()
    right_min = right.min()
    base = max(left_min, right_min)
    return float(signal[idx] - base)


def detect_peaks_1d(signal, smooth_window=9, z_thresh=2.0, min_separation=10, local_win=15):
    x = _rolling_mean(signal, smooth_window)
    n = len(x)
    prev = np.roll(x, 1)
    nextv = np.roll(x, -1)
    cand = (x >= prev) & (x > nextv)
    noise = 1.4826 * _mad(x)
    if noise <= 0:
        noise = 1e-6
    cand_idx = np.where(cand)[0].tolist()
    scored = []
    for i in cand_idx:
        prom = _local_prominence(x, i, local_win)
        z = prom / noise
        if z >= z_thresh:
            scored.append((i, z))
    scored.sort(key=lambda t: t[1], reverse=True)
    keep = []
    taken = np.zeros(n, dtype=bool)
    for i, _ in scored:
        if min_separation <= 0:
            keep.append(i)
            continue
        lo = max(0, i - min_separation)
        hi = min(n, i + min_separation + 1)
        if not taken[lo:hi].any():
            taken[lo:hi] = True
            keep.append(i)
    keep.sort()
    return np.array(keep, dtype=int)


def apply_otsu_mask(img, strength=1.0):
    thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(thr * max(0.5, min(3.0, strength)))
    _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
    return (img.astype(np.float32) * (mask > 0)).astype(np.float32), mask


def sample_circle(cx, cy, r, H, W):
    dtheta = max(1.0 / max(r, 1.0), 2 * np.pi / 4096)
    thetas = np.arange(0, 2 * np.pi, dtheta, dtype=np.float32)
    xs = np.clip(np.rint(cx + r * np.cos(thetas)).astype(int), 0, W - 1)
    ys = np.clip(np.rint(cy + r * np.sin(thetas)).astype(int), 0, H - 1)
    return thetas, xs, ys


def ris_segfree(img_gray,
                use_premask=True, premask_strength=1.0,
                smooth_window=9, z_thresh=2.0, min_sep=10,
                initial_area_pct=10.0, max_area_pct=70.0, steps=5):
    H, W = img_gray.shape
    work = img_gray.copy().astype(np.float32)
    mask_used = None
    if use_premask:
        work, mask_used = apply_otsu_mask(img_gray, strength=premask_strength)
    A = H * W
    r0 = np.sqrt((initial_area_pct / 100.0 * A) / np.pi)
    r1 = np.sqrt((max_area_pct / 100.0 * A) / np.pi)
    radii = np.linspace(r0, r1, steps + 1)
    cx, cy = W / 2.0, H / 2.0
    d_vals, N_vals = [], []
    hits_all = []
    for r in radii:
        thetas, xs, ys = sample_circle(cx, cy, r, H, W)
        profile = work[ys, xs]
        peak_idx = detect_peaks_1d(profile, smooth_window=smooth_window,
                                   z_thresh=z_thresh, min_separation=int(min_sep),
                                   local_win=int(max(10, min(60, r / 2))))
        N = int(len(peak_idx))
        N_vals.append(N)
        d_vals.append(N / (2 * np.pi * r) if r > 0 else 0.0)
        if N > 0:
            sel_theta = thetas[peak_idx]
            hx = cx + r * np.cos(sel_theta)
            hy = cy + r * np.sin(sel_theta)
            hits_all.append(np.stack([hx, hy], axis=1))
    d_vals = np.array(d_vals, dtype=np.float32)
    N_vals = np.array(N_vals, dtype=int)
    hits_xy = np.vstack(hits_all) if hits_all else np.empty((0, 2))
    return radii, N_vals, d_vals, hits_xy, mask_used


