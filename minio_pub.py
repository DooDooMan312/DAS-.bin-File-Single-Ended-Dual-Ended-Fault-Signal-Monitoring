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

# app/minio_pub.py
import os
import json
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


class MinioSender:
    """
    用 MinIO 来“伪装” KafkaSender：
    - 提供 send_json(topic_key, payload, key) / flush()
    - 额外提供 upload_file / upload_dir 方便整目录同步
    """

    def __init__(self):
        self.endpoint   = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket     = os.getenv("MINIO_BUCKET", "das-output")

        # 可选：在所有对象前面加一层前缀，例如 "das"
        self.root_prefix = os.getenv("MINIO_PREFIX", "").strip("/")

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        # bucket 不存在时自动创建（幂等）
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    # ========= Kafka 接口兼容层 =========
    def _build_object_key(self, topic_key: str, key: str | None) -> str:
        """
        把原来的 topic_key / key 映射成 MinIO 里的 object key：
        例如：
          topic_key = "similarity"
          key       = "000000004.bin"
        ->  similarity/000000004.bin_<ts>.json
        最前面再加上 root_prefix（如果设置了 MINIO_PREFIX）。
        """
        # key 里不允许有路径分隔符，简单做个清洗
        if key:
            safe_key = str(key).replace("/", "_")
        else:
            safe_key = "no_key"

        ts_ms = int(time.time() * 1000)
        object_key = f"{topic_key}/{safe_key}_{ts_ms}.json"

        if self.root_prefix:
            object_key = f"{self.root_prefix}/{object_key}"

        return object_key

    def send_json(self, topic_key: str, payload: dict, key: str | None = None):
        """
        以前是发 Kafka，现在改成往 MinIO 写一个 JSON 文件。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        object_key = self._build_object_key(topic_key, key)
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=body,
            ContentType="application/json; charset=utf-8",
        )
        # 你可以加一行 debug，方便在容器日志里看到
        # print(f"[MINIO] put_object bucket={self.bucket} key={object_key}")

    def flush(self, timeout: float = 10):
        """
        Kafka 里是把缓冲区刷出去；MinIO 是 HTTP 直写，没有缓冲，这里就是个空实现。
        保留这个方法是为了兼容 handlers.upload_dir_outputs 里的 sender.flush()。
        """
        return

    # ========= 额外的文件/目录上传工具 =========
    def upload_file(self, local_path: str, object_key: str):
        """上传单个文件到 MinIO"""
        self.client.upload_file(local_path, self.bucket, object_key)

    def upload_dir(self, local_root: Path, prefix: str = ""):
        """
        递归上传整个目录 local_root 到 MinIO：
          MinIO 对象 key = <root_prefix>/<prefix>/<相对路径>
        """
        local_root = Path(local_root).resolve()
        prefix = prefix.strip("/")

        for path in local_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_root)
            key = str(rel).replace("\\", "/")
            if prefix:
                key = f"{prefix}/{key}"
            if self.root_prefix:
                key = f"{self.root_prefix}/{key}"
            self.upload_file(str(path), key)
