# README – Windows 使用说明

本项目提供对 IQ_A / IQ_B 数据目录进行相似度分析、结果输出以及可选的 MinIO 上传功能。
本说明文档用于指导在 Windows 环境下部署、配置和运行本项目。

目录结构示例：

```

├── requirements.txt
├── run_once.py   （调试时使用,不启用小时监听）
├── kafka_pub.py
├── utils_logging.py
├── handlers.py
├── main.py
├── minio_pub.py
├── _minio.py
├── similarity_job.py  
├── utils
│   ├── image_match_v3.py
│   ├── POC.py
│   ├── search_peaks.py
│   ├── bin_to_npy.py
│   ├── create_RP.py
│   ├── frequency_similarity.py
│   └── __init__.py
├── watcher.py
├── pipeline_v4.py
├── config.py
├── config.yaml
└── readme.md

2 directories, 20 files


main.py
pipeline_v4.py
similarity_job.py
watcher.py
config.py
config.yaml
handlers.py
minio_pub.py
_minio.py
utils/
requirements.txt
run_once.py        
```

---

# 1. 环境准备

## 1.1 安装依赖

项目目录下执行：

```
pip install -r requirements.txt
```

---

# 2. 配置文件

本项目读取配置文件 `config.yaml`。
您可以根据需要修改路径、块范围、是否启用 MinIO 等内容。

示例（请按实际情况修改）：

```
io:
  base_dir: "C:\\data\\IQ_A"
  scan_interval_sec: 300
  only_current_hour: false
  ignore_unfinished_hour_on_switch: true
  process_image_bin_as: auto

blocks:
  start: 0
  end: 650

rows:
  max: 1000

minio:
  endpoint: "http://127.0.0.1:9000"
  access_key: "admin"
  secret_key: "minio123456"
  bucket: "das-output"
  enabled: false
```

说明：

* io.base_dir
  A 路数据目录，例如 IQ_A。

* blocks
  起止块范围。

* rows.max
  每个矩阵处理的最大行数。

* minio.enabled
  设置为 false 则不上传结果。

---

# 3. 通过环境变量覆盖配置（可选）

本项目支持通过环境变量覆盖配置文件中的参数，优先级高于 config.yaml。

在 Windows cmd 中示例：

```
set IO_BASE_DIR=D:\lnc_project\Xinwei\1.0.4data\data\IQ_A
set BASE_DIR_2=D:\lnc_project\Xinwei\1.0.4data\data\IQ_B
set NUMBER_ROWS=1000
set NUMBER_BLOCKS=650
set MINIO_ENDPOINT=http://127.0.0.1:9000
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set MINIO_BUCKET=das-output
set CONFIG_FILE=C:\path\to\config.yaml
```

在 PowerShell 中：

```
$env:IO_BASE_DIR="C:\data\IQ_A"
$env:BASE_DIR_2="C:\data\IQ_B"
...
```

运行程序前设置即可。

---

# 4. 运行主程序（监听模式）

主程序入口为 `main.py`。

执行：

```
python main.py
```

程序将执行以下操作：

1. 从 config.yaml 和环境变量读取配置
2. 监听 io.base_dir 目录，每隔 scan_interval_sec 秒扫描一次
3. 当出现新的小时目录（如 2025-11-26 15）时，执行相似度分析
4. 输出结果到当前目录下的 `_out/`
5. 如果开启了 MinIO，将把结果上传到指定 bucket

您可以在控制台看到类似输出：

```
使用块范围: start=0, end=650
使用最大行数 NUMBER_ROWS = 1000
HourWatcher started. base_dir=...
```

---

# 5. 运行单次处理（调试时使用）

如果您只想对某个目录进行单次计算，不启用小时监听，可以使用脚本 `run_once.py`（如项目包含）。

示例：
```
# 使用默认 config.yaml
python run_once.py "C:\data\IQ_A\2025-11-26 15"

# 指定配置文件
python run_once.py "C:\data\IQ_A\2025-11-26 15" -c "C:\path\to\config_win.yaml"

```


程序将：

1. 对该目录进行一次分析
2. 将结果输出到 `_out/`
3. 不进行监听，不循环执行

---

# 6. 输出结果说明

所有处理结果默认输出到 `_out\` 目录中。

典型输出结构：

```
_out
├── double
│   └── 000000000_VS_000000000
├── single
│   ├── 000000000
│   ├── 000000001
│   ├── 000000002
│   ├── 000000003
└── single_file_results
    └── 2025-11-26 16_20251126_161034

```

* all_similarity_summary.csv
  相似度汇总结果

* variance/
  计算的差异矩阵

如果启用了 MinIO，也会同步上传。

---
