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

# app/kafka_pub.py
import logging
from confluent_kafka import Producer
from typing import Dict, Optional
import json

log = logging.getLogger(__name__)

def build_producer_conf(kc: Dict[str, any]) -> Dict[str, any]:
    cfg = {
        "bootstrap.servers": kc["bootstrap_servers"],
        "enable.idempotence": True,     # 你现在默认开着，如果不需要可以改 False
        "socket.keepalive.enable": True,
        "queue.buffering.max.messages": 200000,
    }
    # ---- 基础映射 ----
    if kc.get("acks") is not None: cfg["acks"] = kc["acks"]
    if kc.get("linger_ms") is not None: cfg["linger.ms"] = kc["linger_ms"]
    if kc.get("retries") is not None: cfg["retries"] = kc["retries"]
    if kc.get("max_in_flight_requests_per_connection") is not None:
        cfg["max.in.flight.requests.per.connection"] = kc["max_in_flight_requests_per_connection"]
    if kc.get("compression_type"): cfg["compression.type"] = kc["compression_type"]

    # ---- 安全参数 ----
    if kc.get("security_protocol"): cfg["security.protocol"] = kc["security_protocol"]
    if kc.get("sasl_mechanism"):    cfg["sasl.mechanisms"] = kc["sasl_mechanism"]
    if kc.get("sasl_username"):     cfg["sasl.username"] = kc["sasl_username"]
    if kc.get("sasl_password"):     cfg["sasl.password"] = kc["sasl_password"]
    if kc.get("ssl_ca_location"):   cfg["ssl.ca.location"] = kc["ssl_ca_location"]
    if kc.get("ssl_certificate"):   cfg["ssl.certificate.location"] = kc["ssl_certificate"]
    if kc.get("ssl_key"):           cfg["ssl.key.location"] = kc["ssl_key"]

    # ---- 幂等组合兜底修正（关键）----
    if cfg.get("enable.idempotence"):
        # acks 必须 all
        cfg["acks"] = "all"
        # max.in.flight ≤ 5
        mif = int(cfg.get("max.in.flight.requests.per.connection", 5))
        cfg["max.in.flight.requests.per.connection"] = min(mif, 5)
        # retries > 0
        r = int(cfg.get("retries", 5))
        cfg["retries"] = max(r, 1)

    # 打印一行有效配置帮助排查（不会包含密码）
    safe_cfg = {k: v for k, v in cfg.items() if "password" not in k.lower()}
    log.info("[KAFKA] effective producer conf: %s", safe_cfg)
    return cfg

class KafkaSender:
    def __init__(self, settings):
        self.topics = settings.topics
        self.producer = Producer(build_producer_conf(settings.kafka_conf))

    def send_json(self, topic_key: str, payload: Dict, key: Optional[str] = None):
        topic = self.topics[topic_key]  # 例如 "similarity"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        while True:
            try:
                self.producer.produce(topic=topic, value=data, key=key, callback=self._cb)
                self.producer.poll(0)
                break
            except BufferError:
                self.producer.poll(0.5)

    def _cb(self, err, msg):
        if err:
            print(f"[KAFKA][ERR] {err} for {msg.topic()}")

    def flush(self, timeout: float = 10):
        self.producer.flush(timeout)
