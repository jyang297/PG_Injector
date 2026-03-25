from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterator

from context_manager_config import get_config
from loguru import logger as base_logger


def configure_logger(run_name: str):
    config = get_config()
    log_level = config.observability.log_level
    log_dir = config.observability.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    base_logger.remove()
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    base_logger.add(
        sys.stderr,
        level=log_level,
        format=fmt,
        backtrace=False,
        diagnose=False,
    )
    base_logger.add(
        log_dir / f"{run_name}.log",
        level=log_level,
        format=fmt,
        rotation="10 MB",
        retention=5,
        backtrace=False,
        diagnose=False,
    )
    return base_logger.bind(run_name=run_name)


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def estimate_tokens(text: str) -> int:
    ascii_chars = 0
    cjk_chars = 0
    other_non_ascii = 0
    punctuation_chars = 0

    for char in text:
        if char.isspace():
            continue
        if char.isascii():
            if char.isalnum() or char == "_":
                ascii_chars += 1
            else:
                punctuation_chars += 1
            continue
        if _is_cjk(char):
            cjk_chars += 1
        else:
            other_non_ascii += 1

    return (
        math.ceil(ascii_chars / 4)
        + math.ceil(punctuation_chars / 2)
        + cjk_chars
        + math.ceil(other_non_ascii / 2)
    )


def text_metrics(text: Any) -> dict[str, Any]:
    materialized = "" if text is None else str(text)
    ascii_chars = sum(1 for char in materialized if char.isascii())
    cjk_chars = sum(1 for char in materialized if _is_cjk(char))
    return {
        "chars": len(materialized),
        "utf8_bytes": len(materialized.encode("utf-8")),
        "ascii_chars": ascii_chars,
        "non_ascii_chars": len(materialized) - ascii_chars,
        "cjk_chars": cjk_chars,
        "lines": materialized.count("\n") + 1 if materialized else 0,
        "whitespace_terms": len(materialized.split()),
        "estimated_tokens": estimate_tokens(materialized),
    }


def _json_default(value: Any) -> str:
    return str(value)


def json_payload(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _structure_metrics(payload: Any) -> dict[str, int]:
    counts = {
        "dict_nodes": 0,
        "list_nodes": 0,
        "leaf_nodes": 0,
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            counts["dict_nodes"] += 1
            for child in value.values():
                walk(child)
            return
        if isinstance(value, (list, tuple, set)):
            counts["list_nodes"] += 1
            for child in value:
                walk(child)
            return
        counts["leaf_nodes"] += 1

    walk(payload)
    return counts


def payload_metrics(payload: Any) -> dict[str, Any]:
    # Measure the serialized payload because that is the closest proxy to what a
    # provider actually receives once prompt assembly is done.
    serialized = json_payload(payload)
    summary = text_metrics(serialized)
    top_level_size = len(payload) if hasattr(payload, "__len__") else None
    summary.update(
        {
            "top_level_type": type(payload).__name__,
            "top_level_size": top_level_size,
        }
    )
    summary.update(_structure_metrics(payload))
    return summary


def summarize_text_collection(texts: list[str]) -> dict[str, Any]:
    if not texts:
        return {
            "count": 0,
            "chars_total": 0,
            "chars_avg": 0.0,
            "chars_max": 0,
            "utf8_bytes_total": 0,
            "estimated_tokens_total": 0,
            "estimated_tokens_avg": 0.0,
        }

    metrics = [text_metrics(text) for text in texts]
    total_chars = sum(item["chars"] for item in metrics)
    total_bytes = sum(item["utf8_bytes"] for item in metrics)
    total_tokens = sum(item["estimated_tokens"] for item in metrics)
    return {
        "count": len(texts),
        "chars_total": total_chars,
        "chars_avg": round(total_chars / len(texts), 3),
        "chars_max": max(item["chars"] for item in metrics),
        "utf8_bytes_total": total_bytes,
        "estimated_tokens_total": total_tokens,
        "estimated_tokens_avg": round(total_tokens / len(texts), 3),
    }


def count_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        bucket_key = str(value)
        counts[bucket_key] = counts.get(bucket_key, 0) + 1
    return counts


class RunInstrumentation:
    def __init__(self, run_name: str, logger=None):
        self.run_name = run_name
        self.logger = logger or configure_logger(run_name)
        self.started_at = perf_counter()
        self.metrics: dict[str, Any] = {
            "run_name": run_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
            "texts": {},
            "payloads": {},
        }
        self.logger.info("run_start name={}", run_name)

    @contextmanager
    def stage(self, stage_name: str, **context: Any) -> Iterator[dict[str, Any]]:
        # A plain nested dict keeps the emitted metrics easy to diff, log, and
        # copy into other baseline implementations without extra adapters.
        stage_metrics = self.metrics["stages"].setdefault(stage_name, {})
        if context:
            stage_metrics["context"] = context

        self.logger.info("stage_start name={} context={}", stage_name, context or {})
        started = perf_counter()
        try:
            yield stage_metrics
        except Exception:
            elapsed_ms = round((perf_counter() - started) * 1000, 3)
            stage_metrics["status"] = "error"
            stage_metrics["elapsed_ms"] = elapsed_ms
            self.logger.exception(
                "stage_error name={} elapsed_ms={}",
                stage_name,
                elapsed_ms,
            )
            raise

        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        stage_metrics["status"] = "ok"
        stage_metrics["elapsed_ms"] = elapsed_ms
        self.logger.info("stage_end name={} elapsed_ms={}", stage_name, elapsed_ms)

    def record(self, section: str, value: Any) -> Any:
        self.metrics[section] = value
        self.logger.info("metric section={} value={}", section, value)
        return value

    def record_text(self, label: str, text: Any) -> dict[str, Any]:
        summary = text_metrics(text)
        self.metrics["texts"][label] = summary
        self.logger.info("text_metrics label={} summary={}", label, summary)
        return summary

    def record_payload(self, label: str, payload: Any) -> dict[str, Any]:
        summary = payload_metrics(payload)
        self.metrics["payloads"][label] = summary
        self.logger.info("payload_metrics label={} summary={}", label, summary)
        return summary

    def finalize(self) -> dict[str, Any]:
        total_elapsed_ms = round((perf_counter() - self.started_at) * 1000, 3)
        self.metrics["total_elapsed_ms"] = total_elapsed_ms
        self.logger.info("run_end name={} total_elapsed_ms={}", self.run_name, total_elapsed_ms)
        return self.metrics
