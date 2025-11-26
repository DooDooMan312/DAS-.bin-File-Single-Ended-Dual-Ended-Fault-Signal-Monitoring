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

# app/watcher.py
import os
import re
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Set, Optional
from similarity_job import run_once_for_dir

from config import load_settings

log = logging.getLogger(__name__)

_HOUR_PATTERNS = (
    "%Y-%m-%d_%H",  # 2025-10-22_13
    "%Y-%m-%d %H",  # 2025-10-22 13
    "%Y%m%d_%H",    # 20251022_13
)

def _now_hour_key() -> str:
    """返回当前小时的标准 key（使用第一种格式作为内部标准）。"""
    return datetime.now().strftime(_HOUR_PATTERNS[0])

def _parse_hour_dir_name(name: str) -> Optional[str]:
    """将目录名规范化为标准 key；匹配失败返回 None。"""
    for pat in _HOUR_PATTERNS:
        try:
            dt = datetime.strptime(name, pat)
            return dt.strftime(_HOUR_PATTERNS[0])
        except ValueError:
            continue
    return None

class HourWatcher:
    """
    轮询 base_dir 下的“按小时分目录”的数据。
    - 只处理每个小时目录中新出现/尚未处理的文件
    - 支持只看“当前小时”，或同时扫描历史小时
    - 支持换小时时是否丢弃上一小时未完成的文件集合
    """

    def __init__(
        self,
        base_dir: str,
        interval: float = 5.0,
        only_current_hour: bool = True,
        ignore_on_switch: bool = True,
        stable_secs: float = 3.0,
        include_ext: Optional[Set[str]] = None,
    ):
        self.base_dir = str(base_dir)
        self.interval = float(interval)
        self.only_current_hour = bool(only_current_hour)
        self.ignore_on_switch = bool(ignore_on_switch)
        self.stable_secs = float(stable_secs)
        self.include_ext = {e.lower() for e in (include_ext or {".bin"})}

        self._stop = False
        # 记录每个小时目录已处理的文件集合
        self._processed: Dict[str, Set[str]] = {}
        self._current_key: Optional[str] = None  # 规范化后的 "YYYY-MM-DD_HH"

    # ---------- 信号/控制 ----------
    def _setup_signals(self):
        def _halt(_sig, _frm):
            log.info("Received stop signal; exiting watcher loop...")
            self._stop = True
        signal.signal(signal.SIGTERM, _halt)
        signal.signal(signal.SIGINT, _halt)

    # ---------- 目录/文件发现 ----------
    def _hour_dir_candidates(self) -> List[Path]:
        """返回应扫描的小时目录列表（已按时间排序），取决于 only_current_hour 设置。"""
        base = Path(self.base_dir)
        if not base.is_dir():
            return []
        norm_map: Dict[str, Path] = {}
        for p in base.iterdir():
            if not p.is_dir():
                continue
            norm = _parse_hour_dir_name(p.name)
            if norm:
                norm_map[norm] = p

        if not norm_map:
            return []

        if self.only_current_hour:
            cur_key = _now_hour_key()
            return [norm_map[cur_key]] if cur_key in norm_map else []

        # 否则：按小时排序扫描（旧→新）
        return [norm_map[k] for k in sorted(norm_map.keys())]

    def _list_new_files(self, d: Path, key: str) -> List[str]:
        """列出目录 d 内未处理且稳定的目标扩展文件路径（绝对路径）。"""
        processed = self._processed.setdefault(key, set())

        try:
            children = list(d.iterdir())
        except FileNotFoundError:
            return []

        now = time.time()
        result: List[str] = []
        for c in children:
            if c.is_dir():
                # 跳过输出目录等
                if c.name.startswith("_out"):
                    continue
                continue
            ext = c.suffix.lower()
            if self.include_ext and ext not in self.include_ext:
                continue
            # 文件需“稳定”若干秒，避免生产写入中被抢
            try:
                st = c.stat()
            except FileNotFoundError:
                continue
            if (now - st.st_mtime) < self.stable_secs:
                continue
            abspath = str(c.resolve())
            if abspath not in processed:
                result.append(abspath)
        result.sort()
        return result

    # ---------- 主循环 ----------
    def run(self, on_files: Callable[[str, List[str]], None]):
        """
        on_files(hour_dir: str, files: List[str]) -> None
        - hour_dir: 小时目录的绝对路径
        - files:    新发现的、未处理过的稳定文件（绝对路径）
        """
        self._setup_signals()
        log.info("HourWatcher started. base_dir=%s, interval=%.2fs, only_current_hour=%s, ignore_on_switch=%s",
                 self.base_dir, self.interval, self.only_current_hour, self.ignore_on_switch)

        while not self._stop:
            try:
                hour_dirs = self._hour_dir_candidates()
                cur_key = _now_hour_key()

                # 小时切换：必要时丢弃上一个小时的已处理集合
                if self._current_key != cur_key:
                    if self._current_key is not None and self.ignore_on_switch:
                        # 只清理“上一小时”的处理集合；历史小时的集合保留，避免重复处理
                        if self._current_key in self._processed:
                            log.info("Switch hour → %s, dropping processed set of previous hour %s",
                                     cur_key, self._current_key)
                            self._processed.pop(self._current_key, None)
                    self._current_key = cur_key

                # 没有小时目录也正常等待
                if not hour_dirs:
                    log.debug("No hour directories available under %s", self.base_dir)
                    time.sleep(self.interval)
                    continue

                for d in hour_dirs:
                    key = _parse_hour_dir_name(d.name)
                    if not key:
                        continue
                    new_files = self._list_new_files(d, key)
                    if new_files:
                        log.info("Found %d new files in %s", len(new_files), d)
                        try:
                            on_files(str(d.resolve()), new_files)
                            # 成功后加入已处理集合
                            self._processed[key].update(new_files)
                        except Exception as e:
                            # 不因业务回调异常导致 watcher 停止
                            log.exception("[on_files] raised error: %s", e)

                time.sleep(self.interval)

            except Exception as e:
                # 任何意外都不退出；记录并继续
                log.exception("[watcher] loop error: %s", e)
                time.sleep(self.interval)


# ---------------------
# 直接运行（可选）
# ---------------------
def main():
    settings = load_settings()
    watcher = HourWatcher(
        base_dir=settings.base_dir,
        interval=settings.scan_interval,
        only_current_hour=settings.only_current_hour,
        ignore_on_switch=settings.ignore_on_switch,
        stable_secs=3.0,
        include_ext={".bin"},
    )
    watcher.run(lambda d, files: run_once_for_dir(hour_dir=Path(d), start_block=settings.blocks["start"],
    end_block=settings.blocks["end"],
    max_rows=settings.max_rows,))

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
