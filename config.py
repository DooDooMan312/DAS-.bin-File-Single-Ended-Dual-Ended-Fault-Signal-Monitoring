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

# app/config.py
from pathlib import Path
import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict

# config.py 所在目录，比如 /app/app
BASE_DIR = Path(__file__).resolve().parent
# 默认配置文件 = 同目录下的 config.yaml
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return int(v)

@dataclass
class Settings:
    raw: Dict[str, Any]

    # ---------------- IO ----------------
    @property
    def base_dir(self) -> str:
        # 兼容老变量 BASE_DIR，新变量 IO_BASE_DIR
        return os.getenv("IO_BASE_DIR", os.getenv("BASE_DIR", self.raw["io"]["base_dir"]))

    @property
    def base_dir_2(self) -> str | None:
        """
        B 路根目录：
        优先环境变量 BASE_DIR_2 / SECOND_BASE_DIR，
        否则使用 config.yaml 中 io.base_dir_2（如存在）。
        """
        env_val = os.getenv("BASE_DIR_2") or os.getenv("SECOND_BASE_DIR")
        if env_val:
            return env_val

        io_cfg = self.raw.get("io", {})
        return io_cfg.get("base_dir_2")  # 允许为 None

    class Settings:
        raw: Dict[str, Any]

        # ---------------- IO ----------------
        @property
        def base_dir(self) -> str:
            # 兼容老变量 BASE_DIR，新变量 IO_BASE_DIR
            return os.getenv("IO_BASE_DIR", os.getenv("BASE_DIR", self.raw["io"]["base_dir"]))

        @property
        def base_dir_2(self) -> str | None:
            """
            B 路根目录：
            优先环境变量 BASE_DIR_2 / SECOND_BASE_DIR，
            否则使用 config.yaml 中 io.base_dir_2（如存在）。
            """
            env_val = os.getenv("BASE_DIR_2") or os.getenv("SECOND_BASE_DIR")
            if env_val:
                return env_val

            io_cfg = self.raw.get("io", {})
            return io_cfg.get("base_dir_2")  # 允许为 None

    @property
    def scan_interval(self) -> float:
        # 兼容老变量 POLL_SECONDS，新变量 IO_SCAN_INTERVAL_SEC
        v = os.getenv("IO_SCAN_INTERVAL_SEC", os.getenv("POLL_SECONDS"))
        return float(v if v is not None else self.raw["io"]["scan_interval_sec"])

    @property
    def only_current_hour(self) -> bool:
        # 新变量 IO_ONLY_CURRENT_HOUR，兼容 ONLY_CURRENT_HOUR
        v = os.getenv("IO_ONLY_CURRENT_HOUR", os.getenv("ONLY_CURRENT_HOUR"))
        if v is not None:
            return _env_bool("IO_ONLY_CURRENT_HOUR", _env_bool("ONLY_CURRENT_HOUR", True))
        return bool(self.raw["io"]["only_current_hour"])

    @property
    def ignore_on_switch(self) -> bool:
        # 新变量 IO_IGNORE_ON_SWITCH，兼容 IGNORE_UNFINISHED_HOUR_ON_SWITCH
        v = os.getenv("IO_IGNORE_ON_SWITCH", os.getenv("IGNORE_UNFINISHED_HOUR_ON_SWITCH"))
        if v is not None:
            return _env_bool("IO_IGNORE_ON_SWITCH",
                             _env_bool("IGNORE_UNFINISHED_HOUR_ON_SWITCH", True))
        return bool(self.raw["io"]["ignore_unfinished_hour_on_switch"])

    @property
    def process_image_bin_as(self) -> str:
        return self.raw["io"].get("process_image_bin_as", "auto")

    @property
    def push_images(self) -> bool:
        return _env_bool("PUSH_IMAGES", bool(self.raw.get("push_images", False)))

    @property
    def blocks(self) -> Dict[str, int]:
        """
        统一读块范围：
        - START_BLOCK / END_BLOCK 为老名字
        - NUMBER_BLOCKS 为新名字，只控制 end
        - 如果都没配，就用 config.yaml 里的 blocks.start / blocks.end
        """
        raw_blocks   = self.raw.get("blocks", {})
        default_start = int(raw_blocks.get("start", 0))
        default_end   = int(raw_blocks.get("end", 550))

        # start：ENV 优先，默认用 config
        start = _env_int("START_BLOCK", default_start)

        # end：优先 NUMBER_BLOCKS，其次 END_BLOCK，最后 config
        if os.getenv("NUMBER_BLOCKS") not in (None, ""):
            end = _env_int("NUMBER_BLOCKS", default_end)
        else:
            end = _env_int("END_BLOCK", default_end)

        return {"start": start, "end": end}

    @property
    def max_rows(self):
        """
        控制“每个文件最多用多少行/样本”：
        - NUMBER_ROWS 环境变量优先
        - config.yaml 里可以有 rows.max 作为默认
        - 如果都没有，返回 None 表示不限制
        """
        raw_rows = self.raw.get("rows", {})
        default_max = raw_rows.get("max")  # 允许为 None

        env_val = os.getenv("NUMBER_ROWS")
        if env_val not in (None, ""):
            # 环境变量优先；0 或负数视为“不限制”
            v = _env_int("NUMBER_ROWS", 0)
            return v if v > 0 else None

        if default_max is None:
            return None
        default_max = int(default_max)
        return default_max if default_max > 0 else None


    # ---------------- MinIO ----------------
    @property
    def minio_conf(self) -> Dict[str, Any]:
        m = self.raw.get("minio", {})
        return {
            "endpoint": os.getenv("MINIO_ENDPOINT", m.get("endpoint", "http://minio:9000")),
            "access_key": os.getenv("MINIO_ACCESS_KEY", m.get("access_key", "")),
            "secret_key": os.getenv("MINIO_SECRET_KEY", m.get("secret_key", "")),
            "bucket": os.getenv("MINIO_BUCKET", m.get("bucket", "das-results")),
            "base_prefix": os.getenv("MINIO_BASE_PREFIX", m.get("base_prefix", "")),
        }

    @property
    def minio_prefixes(self) -> Dict[str, str]:
        """
        映射逻辑 topic_key -> 在 MinIO 里的前缀
        比如 similarity -> "similarity/"
        """
        base = self.minio_conf["base_prefix"]
        if base:
            base = base.rstrip("/") + "/"
        return {
            "similarity": base + "similarity/",
            "graph": base + "graph/",
            "peaks": base + "peaks/",
            "alert": base + "alert/",
        }

    # ---------------- Kafka（通用键，给你其他代码读） ----------------
    @property
    def kafka_conf(self) -> Dict[str, Any]:
        # bootstrap 兼容两个名字
        bootstrap = (os.getenv("KAFKA_BOOTSTRAP_SERVERS")
                     or os.getenv("KAFKA_BOOTSTRAP")
                     or self.raw["kafka"]["bootstrap_servers"])
        r = {
            "bootstrap_servers": bootstrap,
            "acks":              os.getenv("KAFKA_ACKS", self.raw["kafka"].get("acks", "all")),
            "linger_ms":         _env_int("KAFKA_LINGER_MS", int(self.raw["kafka"].get("linger_ms", 50))),
            "retries":           _env_int("KAFKA_RETRIES", int(self.raw["kafka"].get("retries", 5))),
            "max_in_flight_requests_per_connection":
                                _env_int("KAFKA_MAX_IN_FLIGHT", int(self.raw["kafka"].get("max_in_flight", 5))),
            "compression_type":  os.getenv("KAFKA_COMPRESSION_TYPE", self.raw["kafka"].get("compression_type", "lz4")),
            "enable_idempotence": _env_bool("KAFKA_ENABLE_IDEMPOTENCE",
                                            bool(self.raw["kafka"].get("enable_idempotence", True))),
            "message_timeout_ms": _env_int("KAFKA_MESSAGE_TIMEOUT_MS", 60000),

            # 安全（全部可用 ENV 覆盖）
            "security_protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", self.raw["kafka"].get("security_protocol")),
            "sasl_mechanism":    os.getenv("KAFKA_SASL_MECHANISM",    self.raw["kafka"].get("sasl_mechanism")),
            "sasl_username":     os.getenv("KAFKA_SASL_USERNAME",     self.raw["kafka"].get("sasl_username")),
            "sasl_password":     os.getenv("KAFKA_SASL_PASSWORD",     self.raw["kafka"].get("sasl_password")),
            "ssl_ca_location":   os.getenv("KAFKA_SSL_CA_LOCATION",   self.raw["kafka"].get("ssl_ca_location")),
            "ssl_certificate":   os.getenv("KAFKA_SSL_CERT_LOCATION", self.raw["kafka"].get("ssl_cert_location")),
            "ssl_key":           os.getenv("KAFKA_SSL_KEY_LOCATION",  self.raw["kafka"].get("ssl_key_location")),
        }
        return r

    # ---------------- confluent-kafka 直接可用的 Producer 配置 ----------------
    def producer_conf(self) -> Dict[str, Any]:
        k = self.kafka_conf
        conf = {
            "bootstrap.servers": k["bootstrap_servers"],
            "acks": k["acks"],
            "linger.ms": k["linger_ms"],
            "retries": k["retries"],
            "compression.type": k["compression_type"],
            "enable.idempotence": k["enable_idempotence"],
            "max.in.flight.requests.per.connection": k["max_in_flight_requests_per_connection"],
            "socket.keepalive.enable": True,
            "queue.buffering.max.messages": 200000,
            "message.timeout.ms": k["message_timeout_ms"],
        }
        # 安全可选项
        if k.get("security_protocol"): conf["security.protocol"] = k["security_protocol"]
        if k.get("sasl_mechanism"):    conf["sasl.mechanism"] = k["sasl_mechanism"]
        if k.get("sasl_username"):     conf["sasl.username"] = k["sasl_username"]
        if k.get("sasl_password"):     conf["sasl.password"] = k["sasl_password"]
        if k.get("ssl_ca_location"):   conf["ssl.ca.location"] = k["ssl_ca_location"]
        if k.get("ssl_certificate"):   conf["ssl.certificate.location"] = k["ssl_certificate"]
        if k.get("ssl_key"):           conf["ssl.key.location"] = k["ssl_key"]
        return conf

    # ---------------- Topics / Thresholds / 其他 ----------------
    @property
    def topics(self) -> Dict[str, str]:
        env_override = {
            "similarity": os.getenv("TOPIC_SIMILARITY"),
            "graph":      os.getenv("TOPIC_GRAPH"),
            "peaks":      os.getenv("TOPIC_PEAKS"),
            "alert":      os.getenv("TOPIC_ALERT"),
        }
        t = dict(self.raw["topics"])
        prefix = os.getenv("TOPIC_PREFIX")
        if prefix:
            t = {k: f"{prefix}.{v.split('.')[-1]}" for k, v in t.items()}
        for k, v in env_override.items():
            if v:
                t[k] = v
        return t

    @property
    def thresholds(self) -> Dict[str, float]:
        return self.raw.get("thresholds", {})

    @property
    def sim_job(self) -> Dict[str, Any]:
        return self.raw.get("similarity_job", {})

def load_settings(path: str | None = None) -> Settings:
    """
    加载配置，优先级：
    1. 显式传入的 path 参数；
    2. 环境变量 CONFIG_FILE；
    3. 默认的 config.yaml（与 config.py 同目录）。

    用法：
        settings = load_settings()                 # 用默认配置
        settings = load_settings("config_win.yaml")  # 指定配置文件
    """
    if path is not None:
        cfg_path = Path(path)
    else:
        env_path = os.getenv("CONFIG_FILE")
        cfg_path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH

    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(raw)

