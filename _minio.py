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

# app/_minio.py
"""
容器内部访问 MinIO 的简单封装 + 自检代码。

依赖环境变量（在 docker-compose 里已经配置）：
  - MINIO_ENDPOINT   如: http://minio:9000
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
  - MINIO_BUCKET     如: das-results
  - MINIO_BASE_PREFIX（可选）如: dev
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


class MinioClient:
    def __init__(self):
        self.endpoint   = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minio123456")
        self.bucket     = os.getenv("MINIO_BUCKET", "das-results")
        self.base_prefix = os.getenv("MINIO_BASE_PREFIX", "").strip().rstrip("/")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )

    # ---- 工具方法 ----
    def _full_key(self, key: str) -> str:
        """在前面自动加 base_prefix（如果设置了）"""
        key = key.lstrip("/")
        if self.base_prefix:
            return f"{self.base_prefix}/{key}"
        return key

    def ensure_bucket(self):
        """如果 bucket 不存在则创建（只用于测试/开发环境）"""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            code = int(e.response.get("Error", {}).get("Code", 0) or 0)
            if code == 404:
                self.s3.create_bucket(Bucket=self.bucket)
            else:
                raise

    def upload_text(self, key: str, text: str, content_type: str = "text/plain; charset=utf-8"):
        key = self._full_key(key)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType=content_type,
        )
        return key

    def upload_json(self, key: str, data: Dict[str, Any]):
        body = json.dumps(data, ensure_ascii=False, indent=2)
        return self.upload_text(key, body, content_type="application/json; charset=utf-8")

    def download_text(self, key: str) -> str:
        key = self._full_key(key)
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read().decode("utf-8")

    def object_exists(self, key: str) -> bool:
        key = self._full_key(key)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_prefix(self, prefix: str):
        prefix = self._full_key(prefix)
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [it["Key"] for it in resp.get("Contents", [])]

    # ---- 自检函数 ----
    def self_test(self):
        """
        在容器内部跑一遍：
          1. 确认能连通 MinIO
          2. 确认 bucket 存在（否则尝试创建）
          3. 上传一个测试 JSON，再读回来验证
        """
        print(f"[minio] endpoint = {self.endpoint}")
        print(f"[minio] bucket   = {self.bucket}")
        if self.base_prefix:
            print(f"[minio] base_prefix = {self.base_prefix}")

        # 1) bucket 是否存在
        print("[minio] checking/creating bucket ...")
        self.ensure_bucket()

        # 2) 写一个测试 JSON
        test_key = "test/minio_self_test.json"
        payload = {"ok": True, "msg": "hello from das-sim container"}
        print(f"[minio] uploading test object: {test_key}")
        full_key = self.upload_json(test_key, payload)
        print(f"[minio] uploaded as: {full_key}")

        # 3) 再读回来
        print(f"[minio] downloading test object: {test_key}")
        text = self.download_text(test_key)
        print("[minio] content:")
        print(text)

        print("[minio] self-test OK")


if __name__ == "__main__":
    client = MinioClient()
    client.self_test()
