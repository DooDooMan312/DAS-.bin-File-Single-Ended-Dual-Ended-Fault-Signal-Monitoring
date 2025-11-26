# -*- coding: utf-8 -*-
"""
DAS 两位置信号相似度评估（整洁版）
- 峰检测与截取
- 频谱可视化与谐波提取
- RP 图相似度（SSIM/NCC/直方图交并）
- 频谱相似度（Welch  cos_psd/coherence/xcorr）
- POC  相位斜率延时（batch）
- 统一保存 CSV（每个 .npy 一个结果表）
"""

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

import os
import time
from typing import List, Dict, Iterable, Optional, Set
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy.typing import ArrayLike
from pathlib import Path
from typing import Optional

# ===== 你自己的工具包 =====
from utils.create_RP import create_RP
from utils.search_peaks import detect_peaks_1d, extract_peak_columns
from utils.image_match_v3 import compute_image_similarity_RP
from utils.frequency_similarity import combined_signal_similarity  # 你之前的频谱相似度函数
from utils.POC import SpecSimParams, batch_events_similarity
from utils.bin_to_npy import convert_bin_dir

# ======================
# 全局配置
# ======================
LIST_DIR = "../bjd9.24/bin"   # 存放 .npy 的目录
OUT_DIR  = "outputs"             # 输出目录（图片与 csv）
os.makedirs(OUT_DIR, exist_ok=True)

# 信号/分析参数（请按实际改）
FS_SIGNAL      = 200.0   # 信号采样率（Hz）——用于 FFT/频谱相似度
ANALYSIS_BAND  = (0.2, 40.0)   # 频谱相似度分析频带
WELCH_NPERSEG  = 32      # 频谱相似度里用的 Welch 参数
WELCH_NOVERLAP = 16

# POC/相位斜率分析参数
POC_FS         = 200.0   # POC/相位斜率分析采用的采样率（建议与 FS_SIGNAL 保持一致）
POC_PARAMS = SpecSimParams(
    fs=POC_FS,
    nperseg=64,          # 短窗建议 64/32
    noverlap=32,
    fmin=1.0,
    fmax=min(40.0, POC_FS/2 - 1e-6),
    use_log_psd=True,
    coh_stat="median",
    freq_scale_search=False,
    s_range=(0.95, 1.05),
    s_steps=31
)

# 峰检测/截取
SMOOTH_WINDOW = 9
SMOOTH_POLY   = 2
PROMINENCE    = 0.7
MIN_DISTANCE  = 120

# FFT 可视化
FFT_ZPF       = 8      # 零填充倍数
FFT_FMAX      = 60.0   # 画图频率上限（Hz）
PEAK_HEIGHT   = 0.05   # 频谱峰检测阈值（按幅度）

# ======================
# 工具函数
# ======================

# ======================
# 核心处理单个文件
# 可配置：目录级输出子目录名
# ======================

OUTPUT_SUBDIR_NAME = "_out"  # 统一放在“根目录/_out”下面


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _global_out_root() -> Path:
    """
    确定 _out 根目录：
      1. 优先使用环境变量 OUT_ROOT（推荐在 Docker 里显式设置）
      2. 其次使用 IO_BASE_DIR / BASE_DIR（和 watcher 的 base_dir 一致）
      3. 最后退回当前工作目录
    """
    root = os.getenv("OUT_ROOT")
    if root:
        base = Path(root)
    else:
        base_dir_env = os.getenv("IO_BASE_DIR") or os.getenv("BASE_DIR")
        if base_dir_env:
            base = Path(base_dir_env)
        else:
            base = Path.cwd()
    base = base.resolve()
    out_root = base / OUTPUT_SUBDIR_NAME
    _ensure_dir(out_root)
    return out_root

def _single_file_outdir(base: str) -> Path:
    """
    单目录模式：
      ${OUT_ROOT}/_out/single/<file1>/
    """
    root = _global_out_root()
    d = root / "single" / base
    _ensure_dir(d)
    return d

def _double_file_outdir(pair_base: str) -> Path:
    """
    双目录模式：
      ${OUT_ROOT}/_out/double/<file1>_VS_<file2>/
    """
    root = _global_out_root()
    d = root / "double" / pair_base
    _ensure_dir(d)
    return d

# 为了兼容老代码，这里定义一个“新的” _new_run_outdir 覆盖前面的版本
def _new_run_outdir(dirpath: str, category: str) -> Path:
    """
    目录级（比如目录方差图和 hourly 汇总表）的输出目录：
      ${OUT_ROOT}/_out/<category>/<hourName>_<timestamp>/
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hour_name = Path(dirpath).name  # 小时目录名，例如 2025-11-08_13
    root = _global_out_root()
    outdir = root / category / f"{hour_name}_{ts}"
    _ensure_dir(outdir)
    return outdir

##########################

##########################
def safe_plot_spectrum(sig: ArrayLike, fs: float, title: str, out_path: str | None = None):
    """窗口化零填充的低频谱图（可选保存）"""
    x = np.asarray(sig, dtype=np.float64)
    N = len(x)
    win = np.hanning(N)
    xw  = (x - x.mean()) * win
    X   = np.fft.rfft(xw, n=N * FFT_ZPF)
    freq = np.fft.rfftfreq(N * FFT_ZPF, d=1.0/fs)
    amp  = (2.0 / (win.sum() / N)) * np.abs(X) / N

    mask = (freq >= 0) & (freq <= FFT_FMAX)
    plt.figure(figsize=(7, 4))
    plt.plot(freq[mask], amp[mask])
    plt.xlim(0, FFT_FMAX)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return freq, amp


def find_harmonics(freq: np.ndarray, amp: np.ndarray, height: float = PEAK_HEIGHT):
    """在谱上找峰——注意这里对的是 amp，不是原始时域 x"""
    peaks, props = find_peaks(amp, height=height)
    return freq[peaks], amp[peaks]


def pack_file_result(base: str,
                     rp_final: float,
                     rp_ssim: float,
                     rp_ncc: float,
                     rp_hist: float,
                     freq_score: float,
                     parts: dict,
                     poc_df_row: pd.Series | None) -> pd.DataFrame:
    """把所有指标打包成一行 DataFrame，便于按文件汇总"""
    row = {
        "file": base,
        "rp_final(%)": rp_final,
        "rp_ssim": rp_ssim,
        "rp_ncc": rp_ncc,
        "rp_hist": rp_hist,
        "freq_score(%)": freq_score,
        "cos_psd_simple": parts.get("cos_psd", np.nan),
        "coherence_simple": parts.get("coherence", np.nan),
        "xcorr_simple": parts.get("xcorr", np.nan),
    }
    if poc_df_row is not None and isinstance(poc_df_row, pd.Series):
        row.update({
            "cos_psd": poc_df_row.get("cos_psd", np.nan),
            "coh_med": poc_df_row.get("coh_med", np.nan),
            "poc_peak": poc_df_row.get("poc_peak", np.nan),
            "delay_phase_ms": poc_df_row.get("delay_phase_ms", np.nan),
            "delay_poc_ms": poc_df_row.get("delay_poc_ms", np.nan),
            "delay_diff_ms": poc_df_row.get("delay_diff_ms", np.nan),
            "phase_delay_r2": poc_df_row.get("phase_delay_r2", np.nan),
        })
    return pd.DataFrame([row])


# ======================
# 核心处理单个文件
# 可配置：目录级输出子目录名
# ======================

OUTPUT_SUBDIR_NAME = "_out"  # 每个数据目录下会生成这个子目录来放所有输出

def _dir_outdir(dirpath: str, subdir_name: str = OUTPUT_SUBDIR_NAME) -> str:
    """
    给定某目录，返回该目录的目录级输出目录：
    <dirpath>/<subdir_name>/
    """
    outdir = os.path.join(os.path.abspath(dirpath), subdir_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _file_outdir(path: str, subdir_name: str = OUTPUT_SUBDIR_NAME) -> Path:
    """
    给定“原始文件路径”或“你希望的基准路径”，生成该文件的输出目录：
    <parent>/<subdir_name>/<file_stem>/
    """
    p = Path(path).resolve()
    base = p.stem
    outdir = p.parent / subdir_name / base
    _ensure_dir(outdir)
    return outdir

def process_single_array(arr2d: np.ndarray, path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    - arr2d: 读自 .bin 的 2D 数组（rows x blocks）或 1D 序列
    - path : 原始文件的绝对/相对路径；决定输出目录名和 base
    """

    p = Path(path).resolve()
    base = p.stem
    outdir = _single_file_outdir(base)     # <dir>/_out/<base>/
    peaks_dir = outdir / "peaks"                        # 峰相关中间件
    spec_dir  = outdir / "spectrum"                     # 频谱图与谐波表
    rp_dir    = outdir / "rp"                           # RP 图产物（create_RP 会用到）
    _ensure_dir(peaks_dir); _ensure_dir(spec_dir); _ensure_dir(rp_dir)

    # 1) 选择 1D 序列：与之前逻辑一致，二维取第 0 行
    y = arr2d.astype(np.float32, copy=False)
    if y.ndim == 2:
        y = y[5, :]

    y = np.asarray(y, dtype=np.float32).ravel()
    if max_rows is not None and max_rows > 0 and len(y) > max_rows:
        y = y[:max_rows]

    # 2) 峰检测
    #todo 本套数据强度都为负值，所以我在两个utils内部增加了取负
    df_peaks = detect_peaks_1d(
        y,
        prominence=PROMINENCE,
        distance=MIN_DISTANCE,
        smooth_window=SMOOTH_WINDOW,
        smooth_poly=SMOOTH_POLY
    )
    (peaks_dir / f"{base}_peaks.csv").write_text(
        df_peaks.to_csv(index=False, encoding="utf-8-sig"),
        encoding="utf-8-sig"
    ) if not df_peaks.empty else None

    if df_peaks.empty:
        # 无峰：仍返回一行空指标，方便汇总
        return pd.DataFrame([{"file": base}])

    # 3) 提取两个峰的局部波形列（y_peaks: shape=(L, 2)）
    y_peaks, x_peaks, idx_df = extract_peak_columns(
        y, df_peaks, show=False,
        outdir=str(peaks_dir)     # 传给你原函数，使用已规范的目录
    )

    # 可选：把峰段保存下来，便于复现（避免反复计算）
    # np.save(str(peaks_dir / f"{base}_y_peaks.npy"), y_peaks)
    # np.save(str(peaks_dir / f"{base}_x_peaks.npy"), x_peaks)
    idx_df.to_csv(peaks_dir / f"{base}_peaks_idx.csv", index=False, encoding="utf-8-sig")

    # 4) 频谱可视化  谐波列表（对两列分别画）
    for col in (0, 1):
        freq, amp = safe_plot_spectrum(
            y_peaks[:, col], FS_SIGNAL,
            title=f"{base} - PeakCol{col} Spectrum",
            out_path=str(spec_dir / f"{base}_spec_col{col}.png")
        )
        h_freqs, h_amps = find_harmonics(freq, amp, height=PEAK_HEIGHT)
        pd.DataFrame({"harmonic_freq": h_freqs, "harmonic_amp": h_amps}) \
          .to_csv(spec_dir / f"{base}_harmonics_col{col}.csv", index=False, encoding="utf-8-sig")

    # 5) RP 图  图像相似度（把 OUTDIR 指向 rp_dir，避免和别的输出混放）
    RP_images = create_RP(y_peaks, eps=0.005, steps=255, OUTDIR=str(rp_dir))
    rp_final, rp_details = compute_image_similarity_RP(
        RP_images[0], RP_images[1], return_details=True
    )
    rp_ssim = rp_details["ssim"]
    rp_ncc  = rp_details["ncc"]
    rp_hist = rp_details["hist"]

    # 6) 频谱相似度
    freq_score, parts = combined_signal_similarity(
        y_peaks[:, 0], y_peaks[:, 1],
        fs=FS_SIGNAL, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP,
        fmin=ANALYSIS_BAND[0], fmax=ANALYSIS_BAND[1]
    )
    seg_a = y_peaks[:, 0]
    seg_b = y_peaks[:, 1]

    # 7) POC  相位斜率（单事件：整段）
    events = [(0, len(y_peaks))]
    poc_df = batch_events_similarity(seg_a, seg_b, events, POC_PARAMS, None)
    poc_df["freq_score"]       = freq_score
    poc_df["cos_psd_simple"]   = parts["cos_psd"]
    poc_df["coherence_simple"] = parts["coherence"]
    poc_df["xcorr_simple"]     = parts["xcorr"]

    # 保存单文件详细结果（事件级）到 <dir>/_out/<base>/similarity/
    sim_dir = outdir / "similarity"
    _ensure_dir(sim_dir)
    poc_df.to_csv(sim_dir / f"{base}_similarity_detail.csv", index=False, encoding="utf-8-sig")

    # 8) 汇总行（返回给上层做目录级/全局汇总）
    file_df = pack_file_result(
        base=base,
        rp_final=rp_final, rp_ssim=rp_ssim, rp_ncc=rp_ncc, rp_hist=rp_hist,
        freq_score=freq_score, parts=parts,
        poc_df_row=poc_df.iloc[0] if not poc_df.empty else None
    )
    return file_df



# ======================
# 新增：从两个不同输入目录各取“最大 prominence 峰”后进行比较
# ======================
def _pick_top_prom_peak_segment(y: np.ndarray, df_peaks: pd.DataFrame) -> np.ndarray:
    """
    在 df_peaks 中按 prominence 最大挑出一个峰，返回该峰的时域片段 (1D)。
    注：extract_peak_columns 已经把 left/right 做了整数化与边界校正。
    """
    if df_peaks.empty:
        raise ValueError("No peaks found.")
    # 取最大 prominence 的行
    row = df_peaks.sort_values("prominence", ascending=False).iloc[0]
    # 构造只含该峰的 DataFrame 以复用 extract_peak_columns
    df_one = pd.DataFrame([row])[["peak_id","left_ip","right_ip","peak_index","prominence","height","width_samples"]]
    y_cols, _, idx_df = extract_peak_columns(y, df_one, plot=False, show=False, outdir=".")
    # y_cols.shape = (L, 1)，取一列
    seg = y_cols[:, 0]
    # 去掉可能的 NaN（理论上已拉伸，无 NaN；此处保险处理）
    return seg[~np.isnan(seg)]


def process_two_input_dirs(dir_a: str, dir_b: str,
                           start_block: int = 0, end_block: int = 550, max_rows: Optional[int] = None) -> Path:
    """
    从两个不同目录里，各自挑出“prominence 最大”的峰段，然后进行相似度评估。
    产物统一落地到：
      ${OUT_ROOT}/_out/double/<fileA>_VS_<fileB>/
    """
    dir_a = Path(dir_a).resolve()
    dir_b = Path(dir_b).resolve()

    # 1) 先把两个目录的 bin -> ndarray
    bin_results_a = convert_bin_dir(
        input_dir=str(dir_a),
        npy_output_dir=None,
        start_block=start_block,
        end_block=end_block,
        make_images=False,
        return_arrays=True,
        # 方差图也一起丢到 pair 目录里
        category_out_root=None,
    )

    bin_results_b = convert_bin_dir(
        input_dir=str(dir_b),
        npy_output_dir=None,
        start_block=start_block,
        end_block=end_block,
        make_images=False,
        return_arrays=True,
        category_out_root=None,
    )

    def _pick_first_valid(recs):
        for r in recs:
            if "error" in r:
                continue
            arr = r.get("selected")
            if arr is None or arr.size == 0:
                continue
            y = arr.astype(np.float32)
            if y.ndim == 2:
                y = y[5, :]
            return r.get("base", "unknown"), y.ravel()
        raise RuntimeError("No valid ndarray in directory")

    base_a, y_a = _pick_first_valid(bin_results_a)
    base_b, y_b = _pick_first_valid(bin_results_b)

    if max_rows is not None and max_rows > 0:
        if y_a.shape[0] > max_rows:
            y_a = y_a[:max_rows]
        if y_b.shape[0] > max_rows:
            y_b = y_b[:max_rows]

    # 2) 各自做峰检测并取“prominence 最大”的那一个峰段
    df_peaks_a = detect_peaks_1d(
        y_a, prominence=PROMINENCE, distance=MIN_DISTANCE,
        smooth_window=SMOOTH_WINDOW, smooth_poly=SMOOTH_POLY
    )
    df_peaks_b = detect_peaks_1d(
        y_b, prominence=PROMINENCE, distance=MIN_DISTANCE,
        smooth_window=SMOOTH_WINDOW, smooth_poly=SMOOTH_POLY
    )
    if df_peaks_a.empty or df_peaks_b.empty:
        raise RuntimeError("No peaks in A or B")

    def _top_prom_segment(y, df_peaks, base_name: str, outdir_for_debug: Path, tag: str):
        # ===建立保存数据的地址===
        top_dir = outdir_for_debug / f"top_{tag}"
        _ensure_dir(top_dir)  # ✅ 关键：确保目录存在

        df1 = (
            df_peaks
            .sort_values("prominence", ascending=False)
            .head(1)
            .reset_index(drop=True)  # ✅ 把这一行加上
        )

        y_peaks, x_peaks, idx_df = extract_peak_columns(
            y, df1,
            plot=True,
            base_name=base_name,
            show=False,
            outdir=str(outdir_for_debug / f"top_{tag}")
        )

        # np.save(str(outdir_for_debug / f"{base_name}_y_peaks.npy"), y_peaks)
        # np.save(str(outdir_for_debug / f"{base_name}_x_peaks.npy"), x_peaks)
        idx_df.to_csv(outdir_for_debug / f"{base_name}_peaks_idx.csv",
                      index=False, encoding="utf-8-sig")

        seg = y_peaks[:, 0]
        return seg[~np.isnan(seg)], idx_df

    # 3) 根据文件名确定 double 的输出目录：
    pair_base = f"{base_a}_VS_{base_b}"
    outdir = _double_file_outdir(pair_base)      # => ${OUT_ROOT}/_out/double/fileA_VS_fileB/
    peaks_dir = outdir / "peaks"
    spec_dir  = outdir / "spectrum"
    rp_dir    = outdir / "rp"
    _ensure_dir(peaks_dir)
    _ensure_dir(spec_dir)
    # _ensure_dir(rp_dir)

    # 这里把 peaks_dir 传进去，同时 base_name 分别用 base_a / base_b，方便区分
    seg_a, idx_a = _top_prom_segment(y_a, df_peaks_a, base_a, peaks_dir, "A")
    seg_b, idx_b = _top_prom_segment(y_b, df_peaks_b, base_b, peaks_dir, "B")
    L = min(len(seg_a), len(seg_b))
    if L < 2:
        raise ValueError(f"两路最大峰长度太短: len(A)={len(seg_a)}, len(B)={len(seg_b)}")
    seg_a = seg_a[:L]
    seg_b = seg_b[:L]

    # 可选：把这俩段保存下来
    # np.save(str(peaks_dir / f"{pair_base}_seg_a.npy"), seg_a)
    # np.save(str(peaks_dir / f"{pair_base}_seg_b.npy"), seg_b)

    # 4) 频谱 + 谐波
    freq_a, amp_a = safe_plot_spectrum(
        seg_a, FS_SIGNAL,
        title=f"{pair_base} - A_topProm",
        out_path=str(spec_dir / f"{pair_base}_A_topProm.png")
    )
    hfa, haa = find_harmonics(freq_a, amp_a, height=PEAK_HEIGHT)
    pd.DataFrame({"harmonic_freq": hfa, "harmonic_amp": haa}) \
      .to_csv(spec_dir / f"{pair_base}_A_harmonics.csv",
              index=False, encoding="utf-8-sig")

    freq_b, amp_b = safe_plot_spectrum(
        seg_b, FS_SIGNAL,
        title=f"{pair_base} - B_topProm",
        out_path=str(spec_dir / f"{pair_base}_B_topProm.png")
    )
    hfb, hab = find_harmonics(freq_b, amp_b, height=PEAK_HEIGHT)
    pd.DataFrame({"harmonic_freq": hfb, "harmonic_amp": hab}) \
      .to_csv(spec_dir / f"{pair_base}_B_harmonics.csv",
              index=False, encoding="utf-8-sig")

    # 5) RP + 图像相似度
    y_peaks = np.stack([seg_a, seg_b], axis=1)  # shape = (L, 2)
    RP_images = create_RP(y_peaks, eps=0.005, steps=255, OUTDIR=str(rp_dir))
    rp_final, rp_details = compute_image_similarity_RP(
        RP_images[0], RP_images[1], return_details=True
    )
    rp_ssim = rp_details["ssim"]
    rp_ncc  = rp_details["ncc"]
    rp_hist = rp_details["hist"]

    # 6) 频谱相似度
    freq_score, parts = combined_signal_similarity(
        seg_a, seg_b,
        fs=FS_SIGNAL,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        fmin=ANALYSIS_BAND[0],
        fmax=ANALYSIS_BAND[1],
    )

    # 7) 事件级 POC（整段视作一个事件）
    events = [(0, L)]

    poc_df = batch_events_similarity(seg_a, seg_b, events, POC_PARAMS)
    poc_df["freq_score"]       = freq_score
    poc_df["cos_psd_simple"]   = parts["cos_psd"]
    poc_df["coherence_simple"] = parts["coherence"]
    poc_df["xcorr_simple"]     = parts["xcorr"]

    sim_dir = outdir / "similarity"
    _ensure_dir(sim_dir)
    poc_df.to_csv(
        sim_dir / f"{pair_base}_similarity_detail.csv",
        index=False, encoding="utf-8-sig"
    )

    # 8) 打包成一行汇总
    poc_row = poc_df.iloc[0] if not poc_df.empty else None
    file_df = pack_file_result(
        base=pair_base,
        rp_final=rp_final, rp_ssim=rp_ssim, rp_ncc=rp_ncc, rp_hist=rp_hist,
        freq_score=freq_score, parts=parts,
        poc_df_row=poc_row,
    )

    # 写一个 pair 级别的汇总 CSV 到同一目录
    summary_path = outdir / "all_similarity_summary.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        new = pd.concat([old, file_df], ignore_index=True)
    else:
        new = file_df
    new.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return outdir


# ----------------------
# 目录级处理（供 watcher/runner 调用）
# ----------------------
def _filter_results_by_files(recs: List[dict], only_files: Optional[Iterable[str]]) -> List[dict]:
    if not only_files:
        return recs
    allow: Set[str] = {Path(f).stem for f in only_files}
    return [r for r in recs if r.get("base") in allow]

def process_hour_dir(
    hour_dir: Path,
    start_block: int = 0,
    end_block: int = 550,
    max_rows: Optional[int] = None,
) -> Path:
    """处理一个小时目录，把产物写到 <hour_dir>/_out/ 并生成目录汇总 CSV。"""
    hour_dir = Path(hour_dir)
    out_root = _new_run_outdir(hour_dir, "single_file_results")

    bin_results = convert_bin_dir(
        input_dir=str(hour_dir),
        npy_output_dir=None,
        start_block=start_block,
        end_block=end_block,
        make_images=True,
        return_arrays=True,
        category_out_root=str(out_root),
    )

    rows: List[pd.DataFrame] = []
    for rec in tqdm(bin_results, desc=f"Process chain in {hour_dir.name}"):
        base = rec.get("base", "unknown")
        if "error" in rec:
            rows.append(pd.DataFrame([{"file": base, "error": rec["error"]}]))
            continue
        arr = rec.get("selected")
        if arr is None:
            rows.append(pd.DataFrame([{"file": base, "error": "missing selected ndarray"}]))
            continue
        try:
            file_df = process_single_array(
                arr,
                path=str(hour_dir / f"{base}.bin"),
                max_rows=max_rows,               # ⭐ 传进去
            )
        except Exception as e:
            file_df = pd.DataFrame([{"file": base, "error": str(e)}])
        rows.append(file_df)

    if rows:
        summary = pd.concat(rows, ignore_index=True)
        (out_root / "all_similarity_summary.csv").write_text(
            summary.to_csv(index=False, encoding="utf-8-sig"),
            encoding="utf-8-sig"
        )
    return out_root

def process_selected_files(
        hour_dir: Path,
        files: List[str],
        start_block: int = 0,
        end_block: int = 550,
        max_rows: Optional[int] = None,
        ) -> Path:
    """只处理传入文件（按 stem 过滤）。"""
    max_rows = max_rows
    hour_dir = Path(hour_dir)
    out_root = _new_run_outdir(hour_dir, "single_file_results")

    bin_results = convert_bin_dir(
        input_dir=str(hour_dir),
        npy_output_dir=None,
        start_block=start_block,
        end_block=end_block,
        make_images=True,
        return_arrays=True,
        image_dir=str(out_root),
    )

    bin_results = _filter_results_by_files(bin_results, files)

    rows: List[pd.DataFrame] = []
    for rec in tqdm(bin_results, desc=f"Process selected in {hour_dir.name}"):
        base = rec.get("base", "unknown")
        if "error" in rec:
            rows.append(pd.DataFrame([{"file": base, "error": rec["error"]}]))
            continue
        arr = rec.get("selected")
        if arr is None:
            rows.append(pd.DataFrame([{"file": base, "error": "missing selected ndarray"}]))
            continue
        try:
            file_df = process_single_array(arr, path=str(hour_dir / f"{base}.bin"), max_rows=max_rows)
        except Exception as e:
            file_df = pd.DataFrame([{"file": base, "error": str(e)}])
        rows.append(file_df)

    if rows:
        summary = pd.concat(rows, ignore_index=True)
        (out_root / "all_similarity_summary.csv").write_text(
            summary.to_csv(index=False, encoding="utf-8-sig"),
            encoding="utf-8-sig"
        )
    return out_root

# ======================
# 主流程：遍历目录（递归），并按“每个目录”各自落地汇总表
# ======================
def main():
    """
    仅处理 .bin：
    - 在 LIST_DIR 下递归查找各目录的 .bin
    - convert_bin_dir 负责读取/切片（内部自带 tqdm，显示“BIN→NPY/Array”）
    - 对每个返回的记录再用 tqdm 做后续处理（process_single_array）
    - 产物落到 <dir>/_out/<base>/...；目录级汇总 <dir>/_out/all_similarity_summary.csv
    """
    if not os.path.isdir(LIST_DIR):
        print(f"[ERROR] LIST_DIR not found: {LIST_DIR}")
        return

    any_bin = False
    dir_summaries: Dict[str, List[pd.DataFrame]] = {}

    for dirpath, _, files in os.walk(LIST_DIR):
        # 找到该目录下的 .bin
        bin_files = sorted([f for f in files if f.lower().endswith(".bin")])
        if not bin_files:
            continue

        any_bin = True
        local_rows: List[pd.DataFrame] = []

        # 目录级输出：把方差图/variance.npy 放在 <dir>/_out/
        dir_out = _dir_outdir(dirpath, OUTPUT_SUBDIR_NAME)

        # 只用内存数组，且（按需）画方差图；不落 .npy
        bin_results = convert_bin_dir(
            input_dir=dirpath,
            npy_output_dir=None,                # 不保存 .npy
            start_block=0,
            end_block=550,
            make_images=True,                   # 方差图/variance.npy → <dir>/_out/
            return_arrays=True,                 # 返回 selected ndarray
            image_dir=dir_out
        )

        # 对每个 bin 结果做后续处理（这里再用 tqdm 展示“处理链”的进度）
        for rec in tqdm(bin_results, desc=f"Process chain in {os.path.basename(dirpath) or dirpath}"):
            base = rec.get("base", "unknown")

            # 若 convert_bin_dir 把错误放进 rec["error"]，这里直接跳过并记录
            if "error" in rec:
                print(f"[SKIP][BIN] {base}: {rec['error']}")
                local_rows.append(pd.DataFrame([{"file": base, "error": rec["error"]}]))
                continue

            try:
                arr = rec.get("selected", None)
                if arr is None:
                    # 可以回退用 out_npy；这里只专注“只用内存”的需求，直接跳过
                    print(f"[SKIP][BIN] {base}: missing 'selected' ndarray in result")
                    local_rows.append(pd.DataFrame([{"file": base, "error": "missing selected ndarray"}]))
                    continue

                # 进入处理链；注意参数名要与函数定义一致
                # 如果需要“方差最大行/均值”策略，可传 trace_strategy="maxvar"/"mean"
                bin_path = os.path.join(dirpath, f"{base}.bin")
                file_df = process_single_array(arr, path=bin_path)

            except Exception as e:
                print(f"[SKIP][BIN] {base}: error -> {e}")
                file_df = pd.DataFrame([{"file": base, "error": str(e)}])

            local_rows.append(file_df)

        # 目录级汇总
        if local_rows:
            summary = pd.concat(local_rows, ignore_index=True)
            save_path = os.path.join(dir_out, "all_similarity_summary.csv")
            summary.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"✅ 目录级汇总已保存: {save_path}")

        dir_summaries[dirpath] = local_rows

    if not any_bin:
        print(f"[WARN] No .bin found under {LIST_DIR}")
        return

    # （可选）全局汇总
    global_rows = [df for rows in dir_summaries.values() for df in rows if rows]
    if global_rows:
        global_summary = pd.concat(global_rows, ignore_index=True)
        global_out = os.path.join(os.path.abspath(LIST_DIR), "_global_out")
        os.makedirs(global_out, exist_ok=True)
        global_csv = os.path.join(global_out, "all_similarity_summary.csv")
        global_summary.to_csv(global_csv, index=False, encoding="utf-8-sig")
        print(f"🌐 全局汇总已保存: {global_csv}")

if __name__ == "__main__":
    dir_a = r'../data/IQ/double/IQ/40/2025-11-12 11/'
    dir_b = r'../data/IQ/double/IQ/45/2025-11-12 11/'
    process_two_input_dirs(dir_a, dir_b,
                               start_block= 0, end_block = 768)
