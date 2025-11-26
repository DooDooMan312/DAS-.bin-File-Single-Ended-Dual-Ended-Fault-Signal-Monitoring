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
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import welch, coherence, get_window
from scipy.fft import rfft, irfft, rfftfreq
from dataclasses import dataclass
import matplotlib.pyplot as plt
import dataclasses

@dataclass
class SpecSimParams:
    fs: float
    nperseg: int = 1024
    noverlap: int = 512
    fmin: float = 0.0
    fmax: float | None = None
    use_log_psd: bool = True
    coh_stat: str = "median"         # "median" or "mean"
    freq_scale_search: bool = False  # 若同源但主频略偏，可设 True
    s_range: tuple[float, float] = (0.9, 1.1)
    s_steps: int = 41

def _welch_psd_z(s: ArrayLike, fs: float, nperseg: int, noverlap: int):
    s = np.asarray(s, dtype=np.float64)
    s = s - np.nanmean(s)
    s = s / (np.nanstd(s) + 1e-12)
    nperseg = int(min(nperseg, len(s)))
    f, P = welch(s, fs=fs, window="hann", nperseg=nperseg,
                 noverlap=noverlap, detrend="constant",
                 return_onesided=True, scaling="density")
    return f, P

def _band_mask(f: np.ndarray, fmin: float, fmax: float):
    fmin = max(0.0, float(fmin))
    fmax = min(float(fmax), float(f[-1]))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        raise ValueError("Empty frequency band after fmin/fmax.")
    return mask

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def phase_only_correlation(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """
    Phase-Only Correlation（POC）
    返回: (poc_peak in [0,1], lag_samples)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(1 << (len(x) + len(y) - 1).bit_length())  # 下一个2次幂以提速
    X = rfft(x, n=n)
    Y = rfft(y, n=n)
    cross = X * np.conj(Y)
    denom = np.abs(cross) + 1e-12
    R = irfft(cross / denom, n=n)  # phase-only correlation
    # 把负滞后部分平移到右侧
    R = np.concatenate([R[-(len(x)-1):], R[:len(y)]])
    lag = np.argmax(R) - (len(x) - 1)
    # 归一化峰值到 [0,1]
    poc_peak = float((np.max(R) - (-1.0)) / (1.0 - (-1.0)))
    return poc_peak, float(lag)

def phase_slope_delay(x: ArrayLike, y: ArrayLike, fs: float,
                      fmin: float, fmax: float) -> tuple[float, float]:
    """
    互谱相位斜率法估计群时延:
    angle(Sxy(f)) ≈ -2π f τ + φ0  →  τ = -slope/(2π)
    返回: (delay_sec, R2_of_fit)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    win = get_window("hann", n, fftbins=True)
    X = rfft((x - x.mean()) * win)
    Y = rfft((y - y.mean()) * win)
    Sxy = X * np.conj(Y)
    f = rfftfreq(n, d=1.0/fs)
    if fmax is None: fmax = f[-1]
    band = (f >= fmin) & (f <= fmax) & (np.abs(Sxy) > 0)
    if not np.any(band):
        return np.nan, 0.0
    ph = np.unwrap(np.angle(Sxy[band]))
    ff = f[band]
    # 可选：按 |Sxy| 加权线性回归
    w = np.abs(Sxy[band])
    w = w / (w.sum() + 1e-12)
    A = np.vstack([2*np.pi*ff, np.ones_like(ff)]).T
    Aw = A * w[:, None]
    yw = ph * w
    # 最小二乘
    theta = np.linalg.lstsq(Aw, yw, rcond=None)[0]  # [slope, intercept]
    slope = theta[0]
    # R^2
    yhat = A @ theta
    ss_res = np.sum((ph - yhat)**2)
    ss_tot = np.sum((ph - np.mean(ph))**2) + 1e-12
    r2 = 1 - ss_res/ss_tot
    tau = -slope  # because ph ≈ -2π f τ + φ0, slope is 2π*τ -> with A using 2πf
    tau = tau / (2*np.pi)
    return float(tau), float(max(0.0, min(1.0, r2)))

def spectral_similarity_with_poc(x: ArrayLike, y: ArrayLike, p: SpecSimParams):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Welch PSD & 频带
    f, Px = _welch_psd_z(x, p.fs, p.nperseg, p.noverlap)
    _, Py = _welch_psd_z(y, p.fs, p.nperseg, p.noverlap)
    fmax = p.fmax if p.fmax is not None else f[-1]
    band = _band_mask(f, p.fmin, fmax)
    f_b = f[band]
    Px_b, Py_b = Px[band], Py[band]

    # cos_psd（对幅值缩放不敏感）
    eps = 1e-12
    A = np.log(Px_b + eps) if p.use_log_psd else Px_b
    B = np.log(Py_b + eps) if p.use_log_psd else Py_b
    if p.freq_scale_search:
        s_grid = np.linspace(p.s_range[0], p.s_range[1], p.s_steps)
        best = -1.0
        for s in s_grid:
            f_s = np.clip(f_b * s, f_b[0], f_b[-1])
            B_s = np.interp(f_b, f_s, B)
            best = max(best, _cos_sim(A, B_s))
        cos_psd = best
    else:
        cos_psd = _cos_sim(A, B)

    # coherence（频带稳健统计）
    f_c, Cxy = coherence(x, y, fs=p.fs, window="hann",
                         nperseg=min(p.nperseg, len(x), len(y)),
                         noverlap=p.noverlap)
    Cxy_b = Cxy[(f_c >= p.fmin) & (f_c <= fmax)]
    coh = float(np.nanmedian(Cxy_b) if p.coh_stat == "median" else np.nanmean(Cxy_b))

    # POC
    poc_peak, poc_lag = phase_only_correlation(x, y)
    poc_delay = poc_lag / p.fs  # seconds

    # 相位斜率延时
    tau_ps, r2_ps = phase_slope_delay(x, y, p.fs, p.fmin, fmax)

    return {
        "cos_psd": float(cos_psd),
        "coherence": float(coh),
        "poc_peak": float(poc_peak),            # ∈[0,1]
        "poc_delay_s": float(poc_delay),        # POC估计的时延
        "phase_delay_s": float(tau_ps),         # 相位斜率法时延
        "phase_delay_r2": float(r2_ps),         # 拟合优度
    }

def tune_params_for_segment(params: SpecSimParams, seg_len: int) -> SpecSimParams:
    """
    根据片段长度自适应 STFT/Welch 参数：
    - 目标：让 Welch 至少产生 3~4 个段（m>=3），短窗情形提升稳健性
    - 对 180 点：会得到 nperseg=64, noverlap=32（≈4 段）
    """
    if seg_len < 64:
        # 太短：还能算 POC/相位斜率，但 Welch/coherence 很不稳
        return dataclasses.replace(params, nperseg=seg_len, noverlap=seg_len//2)

    # 取 <= seg_len/2 的最大 2 的幂，且下限 32，上限 seg_len
    target = max(32, int(2 ** np.floor(np.log2(max(32, seg_len // 2)))))
    nperseg = int(min(target, seg_len))
    noverlap = int(nperseg // 2)  # 50% 重叠通常更稳

    return dataclasses.replace(params, nperseg=nperseg, noverlap=noverlap)

def batch_events_similarity(x: ArrayLike, y: ArrayLike,
                            events: list[tuple[int,int]],
                            params: SpecSimParams,
                            verdict_thresholds: dict | None = None) -> pd.DataFrame:
    if verdict_thresholds is None:
        verdict_thresholds = dict(
            coh_med=0.65,          # 短窗把阈值稍放宽
            cos_psd=0.90,
            poc_peak=0.60,
            delay_consistency_ms=3.0,  # 短窗延时一致性阈值略放宽
            phase_r2=0.50
        )
    rows = []
    for i, (s, e) in enumerate(events, start=1):
        xi = np.asarray(x[s:e])
        yi = np.asarray(y[s:e])
        N = len(xi)
        if N < 32:
            rows.append(dict(event=i, start=s, end=e, note=f"segment extremely short ({N})"))
            continue

        # 针对该事件自适应 Welch 参数
        p_i = tune_params_for_segment(params, N)

        # 计算指标
        try:
            m = spectral_similarity_with_poc(xi, yi, p_i)
        except Exception as err:
            rows.append(dict(event=i, start=s, end=e, note=f"error: {err}"))
            continue

        # 判定（短窗更依赖 POC/相位延时）
        delay_ms = 1e3 * m["phase_delay_s"]
        poc_delay_ms = 1e3 * m["poc_delay_s"]
        delay_diff_ms = float(abs(delay_ms - poc_delay_ms))

        votes = 0
        votes += int(m.get("coherence", 0.0) >= verdict_thresholds["coh_med"])
        votes += int(m["cos_psd"] >= verdict_thresholds["cos_psd"])
        votes += int(m["poc_peak"] >= verdict_thresholds["poc_peak"])
        votes += int(delay_diff_ms <= verdict_thresholds["delay_consistency_ms"])
        votes += int(m["phase_delay_r2"] >= verdict_thresholds["phase_r2"])

        verdict = "同源(强)" if votes >= 4 else ("同源(弱)" if votes == 3 else "非同源/差异大")

        rows.append(dict(
            event=i, start=s, end=e, dur_s = N / params.fs,
            nperseg=p_i.nperseg, noverlap=p_i.noverlap,  # 记录实际用的参数，方便复盘
            cos_psd=m["cos_psd"],
            coh_med=m["coherence"],
            poc_peak=m["poc_peak"],
            delay_phase_ms=delay_ms,
            delay_poc_ms=poc_delay_ms,
            delay_diff_ms=delay_diff_ms,
            phase_delay_r2=m["phase_delay_r2"],
        ))
    return pd.DataFrame(rows)
def quick_plots(df: pd.DataFrame):
    """三个快速可视化：coh/cos/POC；延时一致性；判定饼图"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    ax = axes[0]
    idx = np.arange(len(df))
    ax.plot(idx, df["coh_med"].values, marker="o", label="coherence (median)")
    ax.plot(idx, df["cos_psd"].values, marker="o", label="cos_psd")
    ax.plot(idx, df["poc_peak"].values, marker="o", label="poc_peak")
    ax.set_title("Event metrics")
    ax.set_xlabel("event #")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(idx, df["delay_phase_ms"].values, marker="o", label="phase delay [ms]")
    ax.plot(idx, df["delay_poc_ms"].values, marker="o", label="POC delay [ms]")
    ax.set_title("Delays by method")
    ax.set_xlabel("event #")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    counts = df["verdict"].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    ax.set_title("Verdicts")
    plt.tight_layout()
    plt.show()
# 你的原始数据
# x, y: 两个位置的 DAS 振动时序 (numpy arrays)
# fs: 采样率 (float)
# events: 列表 [(s0, e0), (s1, e1), ...]；若没有，可以根据能量/阈值先粗检

# 例：随便构造几个区间
# events = [(10000, 10512), (20500, 21024), (32000, 32512), ...]

if __name__ == "__main__":
    fs = 200.0
    N = 180

    # 构造 2 段长度均为 180 的事件（索引从 0 开始，到 N 结束）
    x, y = np.random.randn(N), np.random.randn(N)
    events = [(0, N)]                 # 只有一个事件：整个片段
    # 如果你有更长的序列，再按实际位置给 (start, end)

    params = SpecSimParams(
        fs=fs,
        nperseg=64,                   # 适合 180 点的窗口（配合 noverlap=32 -> 4 个Welch段）
        noverlap=32,
        fmin=1.0,                     # 频带要 < fs/2
        fmax=min(40.0, fs/2 - 1e-6),  # 例：40Hz；若 fs 小，自动收紧
        use_log_psd=True,
        coh_stat="median",
        freq_scale_search=False,      # 短窗里先关，后续再按需打开
        s_range=(0.95, 1.05),
        s_steps=31
    )

    # 若你用了我发的“自适应短窗”函数，也可以：params = tune_params_for_segment(params, N)

    df = batch_events_similarity(x, y, fs, events, params)
    print(df.round(3))
    quick_plots(df)   # 注意：请使用带“列存在性判断”的安全版 quick_plots

