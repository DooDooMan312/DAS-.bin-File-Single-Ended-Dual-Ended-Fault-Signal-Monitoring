#  ------------------------------------------------------------------------------
#  Copyright (c) 2025 Chaos
#  All rights reserved.
#  #
#  This software is proprietary and confidential.
#  Licensed exclusively to Shineway Technologies, Inc for internal use only,
#  according to the NDA / agreement signed on 2025.11.26
#  Unauthorized redistribution or disclosure is prohibited.
#  ------------------------------------------------------------------------------
#
#

import numpy as np
from scipy.signal import welch, coherence, fftconvolve
from numpy.typing import ArrayLike

def _cos_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def _welch_psd_z(s, fs, nperseg, noverlap):
    # 去均值、z-score
    s = np.asarray(s, dtype=np.float64)
    s = s - np.nanmean(s)
    s = s / (np.nanstd(s) + 1e-12)
    nperseg = min(nperseg, len(s))
    if nperseg < 2:
        raise ValueError(
            f"信号太短，无法进行 Welch 估计: len(s)={len(s)}, nperseg={nperseg}"
        )

    # ⭐ 关键修复：保证 noverlap < nperseg
    noverlap = int(min(noverlap, nperseg - 1))
    f, P = welch(s, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                 detrend='constant', scaling='density', return_onesided=True)
    return f, P

def _band_mask(f, fmin, fmax):
    fmin = max(0.0, float(fmin))
    fmax = min(float(fmax), float(f[-1]))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        raise ValueError("Empty frequency band after fmin/fmax.")
    return mask

def spectral_similarity(x: ArrayLike, y: ArrayLike, fs: float,
                        nperseg: int = 1024, noverlap: int = 512,
                        fmin: float = 0.0, fmax: float | None = None,
                        use_log_psd: bool = True,
                        coh_stat: str = "median",
                        freq_scale_search: bool = True,  # 开启转速缩放搜索
                        s_range=(0.9, 1.10), s_steps=40):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    max_seg = min(len(x), len(y))
    nperseg = int(min(nperseg, max_seg))
    if nperseg < 2:
        raise ValueError(
            f"信号太短，无法进行 Welch/Coherence 估计: len(x)={len(x)}, len(y)={len(y)}, nperseg={nperseg}"
        )

    # 保证 noverlap < nperseg
    noverlap = int(min(noverlap, nperseg - 1))
    # -----------------------------------------------------

    # PSD（这里会走你已经修过的 _welch_psd_z）
    f, Px = _welch_psd_z(x, fs, nperseg, noverlap)
    f, Py = _welch_psd_z(y, fs, nperseg, noverlap)


    if fmax is None:
        fmax = f[-1]
    mask = _band_mask(f, fmin, fmax)
    f = f[mask]; Px = Px[mask]; Py = Py[mask]

    eps = 1e-12
    A = np.log(Px + eps) if use_log_psd else Px
    B = np.log(Py + eps) if use_log_psd else Py

    # 频率缩放搜索（应对同源不同转速）
    if freq_scale_search:
        s_min, s_max = s_range
        s_grid = np.linspace(s_min, s_max, s_steps)
        # 以 x 为基准，对 y 在频率轴插值为 y(f*s)
        from numpy import interp
        best = -1.0
        for s in s_grid:
            f_s = np.clip(f * s, f[0], f[-1])
            B_s = interp(f, f_s, B)  # 让 y 对齐到 x 的频率网格
            best = max(best, _cos_sim(A, B_s))
        cos_psd = best
    else:
        cos_psd = _cos_sim(A, B)

    # Coherence（频带内稳健统计）
    f_c, Cxy = coherence(x, y, fs=fs, window='hann',
                         nperseg=nperseg,
                         noverlap=noverlap)
    Cxy = Cxy[(f_c >= fmin) & (f_c <= fmax)]
    if coh_stat == "median":
        coh = float(np.nanmedian(Cxy))
    else:
        coh = float(np.nanmean(Cxy))

    # Phase-only correlation 可替代 xcorr（可选）
    # 这里保留你原来的互相关做 0-1 归一
    from scipy.signal import fftconvolve
    x0 = (np.asarray(x) - np.mean(x)) / (np.std(x) + 1e-12)
    y0 = (np.asarray(y) - np.mean(y)) / (np.std(y) + 1e-12)
    z = fftconvolve(x0, y0[::-1], mode='full') / (len(x0) + 1e-12)
    xcorr = float((np.max(z) - (-1.0)) / (1.0 - (-1.0)))  # 映射到[0,1]

    return {"cos_psd": cos_psd, "coherence": coh, "xcorr": xcorr}

def combined_signal_similarity(x, y, fs, w_cos=0.45, w_coh=0.35, w_xcorr=0.20, **kwargs):
    m = spectral_similarity(x, y, fs, **kwargs)
    score_0_1 = w_cos * m["cos_psd"] + w_coh * m["coherence"] + w_xcorr * m["xcorr"]
    return 100.0 * float(np.clip(score_0_1, 0.0, 1.0)), m
if __name__ == "__main__":
    np_array = np.random.randn(2, 180)
    score, parts = combined_signal_similarity(np_array[0], np_array[1], fs=200, nperseg=32, noverlap=16,
                                              fmin=0.2, fmax=40)
    print(score, parts)