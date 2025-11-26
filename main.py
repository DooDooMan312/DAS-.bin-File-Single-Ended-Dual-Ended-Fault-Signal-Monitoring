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

# app/main.py
import logging
from pathlib import Path
from config import load_settings
from minio_pub import MinioSender      # ✅ 改用 MinIO
from handlers import upload_dir_outputs
from similarity_job import run_once_for_dir
from watcher import HourWatcher
import os


log = logging.getLogger(__name__)


OUTPUT_SUBDIR_NAME = "_out"

def debug_local():
    """
    本地调试入口：不启用 HourWatcher，不上传 MinIO，
    只对指定 hour_dir 跑一次 run_once_for_dir，看看到底哪里出问题。
    """
    os.environ["IO_BASE_DIR"] = "/home/liu-liuliu/Postgraduate/Xinwei/11_8/data/1.0.4data/data/IQ_A"
    os.environ["BASE_DIR_2"] = "/home/liu-liuliu/Postgraduate/Xinwei/11_8/data/1.0.4data/data/IQ_B"
    os.environ["NUMBER_ROWS"] = "1000"
    os.environ["NUMBER_BLOCKS"] = "650"
    os.environ["START_BLOCK"] = "0"
    os.environ["END_BLOCK"] = "650"


    # 1) 手动写死你要调试的小时目录（改成你自己的绝对路径）
    hour_dir = Path("/home/liu-liuliu/Postgraduate/Xinwei/11_8/data/1.0.4data/data/IQ_A/2025-11-26 16")

    # 2) 手动设置块范围和最大行数（等价于 config 里的 blocks + NUMBER_ROWS）
    start_block = 0
    end_block   = 650
    max_rows    = 1000   # 或 None 表示不限制

    # 3) 先看看目录里到底有什么
    print("调试目录:", hour_dir)
    if not hour_dir.exists():
        print("❌ 这个目录不存在，请先确认路径是不是写错了")
        return

    files = sorted(hour_dir.iterdir())
    print(f"目录下共有 {len(files)} 个条目：")
    for f in files:
        print("  -", f.name)

    # 4) 调用原来的核心逻辑
    try:
        run_once_for_dir(
            hour_dir,
            only_files=None,
            start_block=start_block,
            end_block=end_block,
            max_rows=max_rows,
        )
        print("✅ run_once_for_dir 执行完成（没有抛出异常）")
    except Exception as e:
        print("❌ run_once_for_dir 抛出了异常：", repr(e))
        raise

def get_global_out_root() -> Path:
    """
    和 pipeline_v4 里的 _global_out_root 逻辑一致：
    优先 OUT_ROOT，其次 IO_BASE_DIR / BASE_DIR，最后当前工作目录。
    """
    root = os.getenv("OUT_ROOT")
    if not root:
        root = os.getenv("IO_BASE_DIR") or os.getenv("BASE_DIR") or os.getcwd()
    base = Path(root).resolve()
    out_root = base / OUTPUT_SUBDIR_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root

def main():
    settings = load_settings()
    # sender = KafkaSender(settings)
    sender = MinioSender()  # ✅ 新增

    # 这里把块范围和行数从配置里拿出来
    blocks   = settings.blocks
    max_rows = settings.max_rows

    log.info("使用块范围: start=%d, end=%d", blocks["start"], blocks["end"])
    log.info("使用最大行数 NUMBER_ROWS = %s", max_rows if max_rows is not None else "不限制")


    # 创建小时监控器
    watcher = HourWatcher(
        base_dir=settings.base_dir,
        interval=settings.scan_interval,
        only_current_hour=settings.only_current_hour,
        ignore_on_switch=settings.ignore_on_switch,
    )

    # 定义回调：当发现新 bin 文件时
    # 仅替换 on_new_files 回调
    def on_new_files(hour_dir: str, files: list[str]):
        log.info("\n####本软件由BJTU提供####\n开始处理目录：%s，共 %d 个新文件\n####本软件由BJTU提供####\n", hour_dir, len(files))
        # run_once_for_dir 内部已“处理并上报”，这里不再重复 upload_dir_outputs


        # 1) 本地计算，相似度结果写到 <OUT_ROOT>/_out/... 里
        run_once_for_dir(
            Path(hour_dir),
            only_files=None,  # 保持原来的逻辑，你有需要再改
            start_block=blocks["start"],
            end_block=blocks["end"],
            max_rows=max_rows,
        )

        # 2) 把 out_root 整个传到 MinIO

        global_out = get_global_out_root()
        #    比如前缀里带上小时目录名，方便甲方按时间浏览
        prefix = f"results/{Path(hour_dir).name}"
        sender.upload_dir(global_out, prefix=prefix)
        log.info("####本软件由BJTU提供####\n✅ %s 数据已处理并上传至 Minio \n####本软件由BJTU提供####\n", hour_dir)

    watcher.run(on_new_files)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

    # debug_local()
