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
# Licensed to 信维科技 for internal internal-use only.
# Redistribution, disclosure, or use for any other project is prohibited.
# Covered by NDA signed on 2025.11.26.

# run_once.py
"""
对单个目录执行一次相似度计算的独立入口。

用法示例：
    python run_once.py "C:\\data\\IQ_A\\2025-11-26 15"

指定配置文件：
    python run_once.py "C:\\data\\IQ_A\\2025-11-26 15" -c "C:\\path\\to\\config_win.yaml"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import load_settings
from similarity_job import run_once_for_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对指定目录执行一次相似度计算（不启用目录监听）。"
    )
    parser.add_argument(
        "hour_dir",
        help="要处理的目录路径，例如：C:\\data\\IQ_A\\2025-11-26 15",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        default=None,
        help="可选，指定配置文件路径。如果不指定，则使用 CONFIG_FILE 环境变量或默认 config.yaml。",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("run_once")

    # 加载配置（支持传入自定义配置路径）
    settings = load_settings(args.config_path)

    blocks = settings.blocks
    max_rows = settings.max_rows

    log.info(
        "使用块范围: start=%d, end=%d",
        blocks["start"],
        blocks["end"],
    )
    log.info(
        "使用最大行数 NUMBER_ROWS = %s",
        max_rows if max_rows is not None else "不限制",
    )

    # 解析并检查目录
    hour_dir = Path(args.hour_dir).expanduser().resolve()
    if not hour_dir.exists():
        raise FileNotFoundError(f"目录不存在: {hour_dir}")
    if not hour_dir.is_dir():
        raise NotADirectoryError(f"不是目录: {hour_dir}")

    log.info("开始处理目录: %s", hour_dir)

    try:
        run_once_for_dir(
            hour_dir,
            only_files=None,               # 如需限制特定文件，可在此传入列表
            start_block=blocks["start"],
            end_block=blocks["end"],
            max_rows=max_rows,
        )
    except Exception as exc:
        log.exception("处理目录时发生异常: %s", exc)
        raise

    log.info("处理完成。输出请查看 _out 目录。")


if __name__ == "__main__":
    main()

