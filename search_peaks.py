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

import os, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths, savgol_filter
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# ======================
# 1D 峰检测
# ======================



def auto_params(y, prominence, distance, SMOOTH_WINDOW = 9, SMOOTH_POLY = 2):
    """根据数据自动给出较稳健的参数（若未显式指定）
    return: 滤波后的 y（用于尺度估计）、prominence、distance
    """
    y = np.asarray(y, float)
    # 轻度平滑用于估计尺度
    win = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    win = min(win, len(y) - (1 - (len(y) % 2)))  # 防越界，保持奇数
    win = max(win, 5 if 5 % 2 == 1 else 7)
    y_s = savgol_filter(y, win, SMOOTH_POLY) if len(y) >= win else y

    # 以 MAD 估计噪声尺度
    mad = np.median(np.abs(y_s - np.median(y_s)))
    scale = 1.4826 * mad

    if prominence is None:
        # 取 3*噪声尺度，并不超过幅度跨度的一半
        span = np.max(y_s) - np.min(y_s)
        prom = 3 * scale if scale > 0 else 0.1 * span
        prom = min(prom, 0.5 * span)
        prominence = max(prom, 1e-12)

    if distance is None:
        # 让最多能找到 ~30 个峰
        distance = max(1, len(y) // 30)

    return y_s, prominence, distance

def detect_peaks_1d(y, prominence=None, distance=None,
                    smooth_window=9, smooth_poly=2):
    """返回包含所有峰信息的 DataFrame（正峰）"""
    y = np.asarray(y)
    y = np.ravel(y)
    if np.iscomplexobj(y):
        y = np.abs(y)

    # ===== 新增：自动清理 NaN / inf =====
    if not np.all(np.isfinite(y)):
        nan_count = np.sum(~np.isfinite(y))
        print(f"[WARN] detect_peaks_1d: input contains {nan_count} non-finite values; replacing with 0.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    # ====================================

    win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    if len(y) >= win and win >= smooth_poly + 2:
        y_smooth = savgol_filter(y, win, smooth_poly)
    else:
        y_smooth = y

    # 自动参数（在平滑后的序列上估计尺度）
    _ys, prominence, distance = auto_params(y_smooth, prominence, distance)

    # 正峰
    peaks_pos, props_pos = find_peaks(y_smooth, prominence=prominence, distance=distance)

    # ✅ 正确的返回顺序：widths, h_eval, left_ips, right_ips
    widths_pos, h_eval, left_ips_pos, right_ips_pos = peak_widths(y_smooth, peaks_pos, rel_height=0.9)

    rows = []
    for j, p in enumerate(peaks_pos, start=1):
        rows.append({
            "peak_id": int(j),
            "peak_index": int(p),                                  # 峰的位置（索引）
            "height": float(y[p]),                                 # 用原序列 y 的峰高
            "prominence": float(props_pos["prominences"][j - 1]),  # 显著性
            "width_samples": float(widths_pos[j - 1]),             # 半高全宽（样本数）
            "left_ip": float(left_ips_pos[j - 1]),                 # 左交点（浮点索引）
            "right_ip": float(right_ips_pos[j - 1]),               # 右交点（浮点索引）
        })

    df = pd.DataFrame(rows)
    return df

# ======================
# 将每个峰切成列
# ======================
def extract_peak_columns(
    y, df_peaks,
    plot=True,
    base_name="series",
    outdir="outputs",
    show=False,           # 是否 plt.show()；批处理建议 False
    dpi=150,
    stretch_to_max=True,  # 新增：是否把每个峰拉伸到最长长度
):
    """
    y: 1D numpy array
    df_peaks: 包含 ['peak_id','left_ip','right_ip'] 的 DataFrame（left/right 为浮点索引）
    返回:
      y_peaks: shape = (max_len, n_peaks)，每列一个峰，右侧 NaN 补齐
      x_peaks: 同形状，每列为该峰对应的 x 索引（含 NaN）
      idx_df : 仅保留 peak_id, left_i, right_i（整数且已校正）
    """
    y = np.nan_to_num(np.asarray(y), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    n = len(y)
    if df_peaks.empty:
        return (np.empty((0, 0)), np.empty((0, 0)), pd.DataFrame(columns=["peak_id","left_i","right_i"]))

    # ✅ 确保输出目录存在（后面所有 np.save / plt.savefig 都安全）
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    df = df_peaks.copy()

    # 浮点索引 -> 整数索引（包含半高交点）
    df['left_i']  = np.clip(np.floor(df['left_ip']).astype(int), 0, n - 1)
    df['right_i'] = np.clip(np.ceil(df['right_ip']).astype(int),  0, n - 1)

    # 确保 left <= right
    lr = np.sort(df[['left_i','right_i']].to_numpy(), axis=1)
    df[['left_i','right_i']] = lr

    lengths = df['right_i'] - df['left_i'] + 1
    max_len = int(lengths.max())
    k = len(df)

    y_peaks = np.full((max_len, k), np.nan, dtype=float)
    x_peaks = np.full((max_len, k), np.nan, dtype=float)

    for col, row in enumerate(df.itertuples(index=False)):
        l, r = int(row.left_i), int(row.right_i)
        seg_y = y[l:r+1]
        L = len(seg_y)
        y_peaks[:L, col] = seg_y
        x_peaks[:L, col] = np.arange(l, r+1, dtype=float)

    idx_df = df[['peak_id','peak_index','left_i','right_i']].reset_index(drop=True)

    # ====== 新增：把每列拉伸到 max_len（线性重采样，不引入 NaN） ======
    y_peaks_stretched = np.empty_like(y_peaks)
    x_peaks_stretched = np.empty_like(x_peaks)

    for col in range(k):
        # 当前列有效段
        valid = ~np.isnan(y_peaks[:, col])
        cnt = int(valid.sum())
        if cnt <= 0:
            # 保险：整列无效，填 NaN
            y_peaks_stretched[:, col] = np.nan
            x_peaks_stretched[:, col] = np.nan
            continue
        # 取有效 x/y
        xv = x_peaks[valid, col]
        yv = y_peaks[valid, col]

        if cnt == 1:
            # 只有一个点：常值拉伸
            y_peaks_stretched[:, col] = yv[0]
            x_peaks_stretched[:, col] = np.linspace(xv[0], xv[0], max_len)
        else:
            # 目标长度的等距 x（在原 x 区间内）
            x_new = np.linspace(xv[0], xv[-1], max_len)
            # 线性插值；边界都在区间内，不会外推
            y_new = np.interp(x_new, xv, yv)
            y_peaks_stretched[:, col] = y_new
            x_peaks_stretched[:, col] = x_new

    # np.save(os.path.join(outdir, f"y_peaks_stretched.npy"), y_peaks_stretched)

    # ================= 绘图（可选） =================
    if plot:
        os.makedirs(outdir, exist_ok=True)

        # 1) 总览图：整条 y 背景 + 各峰片段高亮
        fig = plt.figure(figsize=(10, 4))
        plt.plot(np.arange(n), y, linewidth=1)  # 背景：整条 y
        for col in range(k):
            # 有效范围
            valid = ~np.isnan(x_peaks[:, col]) & ~np.isnan(y_peaks[:, col])
            if not np.any(valid):
                continue
            # 用 iloc 以“第 col 行”取值
            row_idx = idx_df.iloc[col]
            l_i = int(row_idx['left_i'])
            r_i = int(row_idx['right_i'])

            row_peak = df.iloc[col]
            peak_center = int(row_peak['peak_index'])

            plt.plot(x_peaks[valid, col], y_peaks[valid, col], linewidth=2)
            plt.vlines(l_i, np.min(y), np.max(y), linestyles="--", linewidth=1)
            plt.vlines(r_i, np.min(y), np.max(y), linestyles="--", linewidth=1)
            plt.axvline(peak_center, linestyle=":", linewidth=1)

        plt.title(f"Peaks Overview: {base_name}")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        out_png_overview = os.path.join(outdir, f"{base_name}_peaks_overview.png")
        plt.savefig(out_png_overview, dpi=dpi)
        if show: plt.show()
        plt.close(fig)

        # 2) 单峰图：每列单独保存
        for col in range(k):
            valid = ~np.isnan(x_peaks[:, col]) & ~np.isnan(y_peaks[:, col])
            if not np.any(valid):
                continue

            row_idx = idx_df.iloc[col]
            l_i = int(row_idx['left_i'])
            r_i = int(row_idx['right_i'])

            row_peak = df.iloc[col]
            pid = int(row_peak['peak_id'])
            peak_center = int(row_peak['peak_index'])

            fig = plt.figure(figsize=(8, 3.5))
            plt.plot(np.arange(n), y, linewidth=1)
            plt.plot(x_peaks[valid, col], y_peaks[valid, col], linewidth=2)
            plt.vlines(l_i, np.min(y), np.max(y), linestyles="--", linewidth=1)
            plt.vlines(r_i, np.min(y), np.max(y), linestyles="--", linewidth=1)
            plt.axvline(peak_center, linestyle=":", linewidth=1)

            plt.title(f"{base_name} - Peak {pid} (center={peak_center}, [{l_i},{r_i}])")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            out_png_single = os.path.join(outdir, f"{base_name}_peak_{pid}.png")
            plt.savefig(out_png_single, dpi=dpi)
            if show: plt.show()
            plt.close(fig)

    # return y_peaks, x_peaks, idx_df
    return y_peaks_stretched, x_peaks_stretched, idx_df

if __name__ == "__main__":
    listpath = "../bjd9.24/npy_2D"  # 存放 .npy 的目录
    sp_res = 2.0  # 每个采样点对应的米数（如需物理宽度可用）
    SMOOTH_WINDOW = 9  # SG 平滑窗口（奇数）
    SMOOTH_POLY = 2  # SG 多项式阶数
    PROMINENCE = None  # 峰的突出度阈值；None 表示自动估计
    MIN_DISTANCE = None  # 相邻峰最小间距（样本数）；None 自动估计
    ALIGN_PEAK = True  # 计算相似度时是否按峰中心对齐
    TARGET_LEN = 128  # 相似度计算重采样目标长度
    OUTDIR = "outputs"  # 输出目录（图片与csv）
    os.makedirs(OUTDIR, exist_ok=True)
