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
import numpy as np
# import configparser
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import re
from  config import load_settings
matplotlib.use("Agg")
PARAMS_DEFAULT_FILE = "/data/IQ/params.json"


def _file_outdir_by_base(parent_dir: str, base: str) -> Path:
    d = Path(parent_dir).resolve() / OUTPUT_SUBDIR_NAME / base
    ensure_dir(d)
    return d
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sanitize_base(base: str) -> str:
    # 只保留常见安全字符，其它替换为下划线
    return re.sub(r'[^A-Za-z0-9._-]', '_', base)

def _get_rows_cols_from_env() -> tuple[int, int]:
    """
    优先从环境变量 NUMBER_ROWS / NUMBER_BLOCKS 读取矩阵尺寸；
    如果环境变量缺失，则尝试从 config.yaml (Settings) 中获取：
      - rows.max   -> number_rows
      - blocks.end -> number_blocks
    如果仍然无法确定，则抛出 RuntimeError 提示。
    """
    rows_env = os.getenv("NUMBER_ROWS")
    blocks_env = os.getenv("NUMBER_BLOCKS")

    if rows_env and blocks_env:
        return int(rows_env), int(blocks_env)

    # 环境变量缺失，尝试从 config.yaml 读取
    try:
        settings = load_settings()

        # Settings.max_rows 目前已经表示“NUMBER_ROWS”的含义
        number_rows = settings.max_rows
        # Settings.blocks["end"] 可以作为列数的默认值
        number_blocks = settings.blocks["end"]

        if number_rows is None or number_blocks is None:
            raise ValueError("配置中缺少 rows.max 或 blocks.end")

        return int(number_rows), int(number_blocks)

    except Exception as e:
        raise RuntimeError(
            "缺少矩阵尺寸：请在环境变量（NUMBER_ROWS / NUMBER_BLOCKS）"
            "或 config.yaml（rows.max / blocks.end）中配置矩阵尺寸。例如 NUMBER_ROWS=512, NUMBER_BLOCKS=600。"
        ) from e



def _int_env(name: str, default: Optional[int]=None) -> Optional[int]:
    v = os.getenv(name)
    if v is None or str(v).strip()=="":
        return default
    return int(v)
def _load_params_from_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_matrix_shape() -> tuple[int, int]:
    """
    优先级：环境变量 > /data/IQ/params.json > 报错
      - NUMBER_ROWS / NUMBER_BLOCKS
      - 或 params.json: {"number_rows":..., "number_blocks":...}
    """


    # 1) env
    env_rows = _int_env("NUMBER_ROWS")
    env_blocks = _int_env("NUMBER_BLOCKS")

    if env_rows is not None and env_blocks is not None:

        return int(env_rows), int(env_blocks)
  # 2) mounted json
    params_file = os.getenv("PARAMS_FILE", PARAMS_DEFAULT_FILE)
    data = _load_params_from_json(params_file)

    try:
        return int(data["number_rows"]), int(data["number_blocks"])
    except Exception:
            raise FileNotFoundError(
                f"缺少矩阵尺寸：请设置 env NUMBER_ROWS/NUMBER_BLOCKS，或在 {params_file} 提供 number_rows/number_blocks。")

def _load_matrix(bin_path: str, number_rows: int, number_blocks: int) -> np.ndarray:
    with open(bin_path, "rb") as f:
        data = np.abs(np.fromfile(f, dtype=np.float32)).reshape((number_rows, number_blocks))
    return data

def _compute_variance(selected_blocks: np.ndarray) -> np.ndarray:
    # 你原先现在使用的整体方差（按列）
    return np.var(selected_blocks, axis=0)

def convert_bin_dir(
    input_dir: str,
    npy_output_dir: Optional[str] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    make_images: bool = True,
    return_arrays: bool = True,
    image_dir: Optional[str] = None,
    category_out_root: Optional[str] = None,   # 🆕 新增参数
) -> List[Dict[str, Any]]:
    """
    扫描 input_dir 中的 .bin，并（可选）保存为 .npy、画方差图、返回内存数组。
    返回列表，每个元素字典包含：
      - base: 文件名去扩展（已规范化）
      - selected: 2D ndarray（当 return_arrays=True）
      - variance: 1D ndarray
      - out_npy: 保存路径（当 npy_output_dir 不为 None）
      - image_png / variance_npy: 可选图片/方差npy路径
      - src_bin: 原 .bin 路径
    """
    input_dir_p = Path(input_dir).resolve()
    npy_output_dir_p = Path(npy_output_dir).resolve() if npy_output_dir else None
    image_dir_p = Path(image_dir).resolve() if image_dir else None
    category_root_p = Path(category_out_root).resolve() if category_out_root else None

    # ✅ 优先使用 category_out_root
    if category_out_root:
        category_root = Path(category_out_root).resolve()
        variance_dir = category_root / "variance"
        ensure_dir(variance_dir)
    elif image_dir_p:
        ensure_dir(image_dir_p)
        variance_dir = image_dir_p / "variance"
        ensure_dir(variance_dir)
    else:
        variance_dir = None

    if npy_output_dir_p:
        ensure_dir(npy_output_dir_p)


    bin_files = [f for f in input_dir_p.iterdir() if f.suffix.lower() == ".bin"]
    if not bin_files:
        raise FileNotFoundError(f"在目录 {input_dir_p} 中未找到任何 .bin 文件")

    ##============================
    # rows and blocks 设置
    ##============================

    # ✅ 1. 统一获取矩阵尺寸：建议只用 env（简单明确）
    number_rows, number_blocks = _get_rows_cols_from_env()
    # 如果你想支持 params.json，而不是强制 env，就改成：
    # number_rows, number_blocks = _get_matrix_shape()

    # ✅ 2. 统一的 block 范围优先级：
    #    函数参数 > 环境变量 START_BLOCK/END_BLOCK > 默认 0:number_blocks

    def _fallback_int(name: str, default: Optional[int]) -> Optional[int]:
        # 先看 env
        v_env = _int_env(name.upper(), None)
        if v_env is not None:
            return v_env
        # 不想用 params.json 的话，可以直接 return default
        # 下面这两行可以删掉
        v_json = _load_params_from_json(os.getenv("PARAMS_FILE", PARAMS_DEFAULT_FILE)).get(name.lower(), None)
        return int(v_json) if v_json is not None else default

    if start_block is None:
        start_block = _fallback_int("start_block", 0)

    if end_block is None:
        end_block = _fallback_int("end_block", number_blocks)

    # ✅ 3. 检查合法性
    if start_block < 0 or end_block > number_blocks or start_block >= end_block:
        raise ValueError(f"无效的 Block 范围: {start_block}:{end_block}, 总 Blocks={number_blocks}")
    

    def _fallback_int(k: str, default: Optional[int]) -> Optional[int]:
        v_env = _int_env(k.upper(), None)

    
        if v_env is not None:
            return v_env
        v_json = _load_params_from_json(os.getenv("PARAMS_FILE", PARAMS_DEFAULT_FILE)).get(k, None)
    
        return int(v_json) if v_json is not None else default
    
    
    if start_block is None: start_block = _fallback_int("start_block", 0)
    
    if end_block is None: end_block = _fallback_int("end_block", number_blocks)

    if start_block < 0 or end_block > number_blocks or start_block >= end_block:
        raise ValueError(f"无效的 Block 范围: {start_block}:{end_block}, 总 Blocks={number_blocks}")

    results: List[Dict[str, Any]] = []

    for bin_path in tqdm(sorted(bin_files), desc="BIN→NPY/Array"):
        """
            将.bin文件转为npy
        """
        base_raw = bin_path.stem # 提取文件名
        base = sanitize_base(base_raw)  # 规范化
        rec: Dict[str, Any] = {"base": base, "src_bin": str(bin_path)}
        # 构建名称-路径的dict

        if category_root_p:
            # 将 variance 输出到单次运行的分类目录中
            variance_dir = category_root_p / "variance"
            ensure_dir(variance_dir)

        try:
            matrix = _load_matrix(str(bin_path), number_rows, number_blocks)
            # print("has_nan:", np.isnan(matrix).any())
            # print("has_inf:", np.isinf(matrix).any())

            matrix = np.nan_to_num(matrix)
            if matrix.shape != (number_rows, number_blocks):
                raise ValueError(
                    f"{base}: reshape got {matrix.shape}, expect {(number_rows, number_blocks)}"
                )

            selected = matrix[:, start_block:end_block]
            # print("has_nan:", np.isnan(selected).any())
            # print("has_inf:", np.isinf(selected).any())
            if selected.size == 0:
                raise ValueError(f"{base}: empty slice {start_block}:{end_block}")

            # variance = _compute_variance(selected)
            # rec["variance"] = variance
            #
            # # —— 保存 NPY（可选）——
            # if npy_output_dir_p:
            #     out_npy_path = npy_output_dir_p / f"{base}.npy"
            #     ensure_dir(out_npy_path.parent)
            #     np.save(str(out_npy_path), selected)
            #     rec["out_npy"] = str(out_npy_path)
            #
            # # —— 方差图/方差npy（可选）——
            # if make_images and variance_dir:
            #     png_path = variance_dir / f"{base}_variance.png"
            #     var_npy  = variance_dir / f"{base}_variance.npy"
            #     ensure_dir(png_path.parent)
            #     plt.figure(dpi=200)
            #     plt.plot(variance)
            #     plt.tight_layout()
            #     plt.savefig(str(png_path))
            #     plt.close()
            #     np.save(str(var_npy), variance)
            #     rec["image_png"] = str(png_path)
            #     rec["variance_npy"] = str(var_npy)

            # —— 返回内存数组（可选）——
            if return_arrays:
                rec["selected"] = selected.astype(np.float32, copy=False)

        except Exception as e:
            rec["error"] = str(e)

        results.append(rec)

    return results


# ===main===
if __name__ == "__main__":
    input_dir = r'../../data/IQ/double/IQ/40/2025-11-12 11/'
    npy_output_dir = r'../../test/output'
    start_block = 0
    end_block = None
    image_dir = r'../../test/output'
    function =  convert_bin_dir(
        input_dir,
        npy_output_dir,
        start_block,
        end_block,
        make_images = True,
        return_arrays = False,
        image_dir = image_dir)