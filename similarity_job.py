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
#
#

import os
from config import load_settings, Settings
from typing import Optional, List
from pathlib import Path

import logging
from pipeline_v4 import (
    process_hour_dir as _process_hour_dir,
    process_selected_files,
    process_two_input_dirs,
)
from utils.bin_to_npy import convert_bin_dir

log = logging.getLogger(__name__)

def _resolve_pair_dir(settings: Settings, hour_dir: Path) -> Optional[Path]:
    """
    根据 settings.base_dir  环境变量 BASE_DIR_2/SECOND_BASE_DIR，
    把 A 的小时目录映射到 B 的对应小时目录。

    例如：
      settings.base_dir = /data/IQ_A
      BASE_DIR_2        = /data/IQ_B
      hour_dir          = /data/IQ_A/2025-11-08_13
    则映射为：
      /data/IQ_B/2025-11-08_13
    """
    base_b = settings.base_dir_2
    if not base_b:
        # 未配置 B 路，保持单通路模式
        return None

    root_a = Path(getattr(settings, "base_dir", hour_dir.parent)).resolve()
    h = Path(hour_dir).resolve()
    try:
        rel = h.relative_to(root_a)   # 相对 A 根路径的相对路径，例如 "2025-11-08_13"
    except ValueError:
        # 不在 base_dir 下，就只用目录名兜底
        rel = Path(h.name)

    return Path(base_b).resolve() / rel

def run_once_for_dir(
    hour_dir: Path,
    only_files: Optional[List[str]] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Path:
    """
    处理一个小时目录（可选只处理给定文件），并负责把 _out 里的各类数据上报 Kafka。

    参数优先级：
    - start_block / end_block / max_rows 如果函数参数给了，就用参数
    - 否则，从 config.Settings.blocks / Settings.max_rows 读取（包括环境变量覆盖）
    """
    settings = load_settings()
    hour_dir = Path(hour_dir)

    # ---- 统一决定本次运行要用的块范围 / 行数 ----
    blocks = settings.blocks
    eff_start = blocks["start"] if start_block is None else int(start_block)
    eff_end   = blocks["end"]   if end_block   is None else int(end_block)

    eff_max_rows = settings.max_rows if max_rows is None else max_rows

    log.info(
        "run_once_for_dir: dir=%s, only_files=%s, start_block=%d, end_block=%d, max_rows=%s",
        hour_dir, only_files, eff_start, eff_end,
        eff_max_rows if eff_max_rows is not None else "不限制",
    )

    # ---------- 1. 先跑单目录处理：保证每个文件都有自己的 _out/<base> ----------
    if only_files:
        log.info("Run single-dir selected mode. dir=%s, files=%s", hour_dir, only_files)
        out_root = process_selected_files(
            hour_dir,
            only_files,
            start_block=eff_start,
            end_block=eff_end,
            max_rows=eff_max_rows,   # ⭐ 新增参数，下面你要在函数签名里加
        )
    else:
        log.info("Run single-dir full mode. dir=%s", hour_dir)
        out_root = _process_hour_dir(
            hour_dir,
            start_block=eff_start,
            end_block=eff_end,
            max_rows=eff_max_rows,   # ⭐ 同上
        )

    # ---------- 2. 再尝试跑双目录模式（如果配对目录存在） ----------
    pair_dir = _resolve_pair_dir(settings, hour_dir)
    if pair_dir is not None:
        pair_dir = pair_dir.resolve()
        if pair_dir.exists():
            log.info("Run pair mode. A=%s, B=%s", hour_dir, pair_dir)
            try:
                process_two_input_dirs(
                    str(hour_dir),
                    str(pair_dir),
                    start_block=eff_start,
                    end_block=eff_end,
                    max_rows=eff_max_rows,   # ⭐ 记得在 pipeline_v4.process_two_input_dirs 里加这个参数
                )
            except Exception as e:
                log.error("Pair mode failed for A=%s, B=%s, err=%s", hour_dir, pair_dir, e)
        else:
            log.warning(
                "Pair dir configured but not found for %s: %s; only single-dir results will be generated.",
                hour_dir, pair_dir
            )

    return out_root
