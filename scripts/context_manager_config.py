from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str
    default_resource_owner: str
    default_resource_namespace: str


@dataclass(frozen=True)
class LoaderConfig:
    value_cardinality_limit: int
    data_dir: Path
    source_adapter: str


@dataclass(frozen=True)
class QueryConfig:
    hybrid_search_limit: int
    top_candidate_columns: int
    chunk_type_weights: Mapping[str, float]


@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: str
    log_dir: Path


@dataclass(frozen=True)
class ContextManagerConfig:
    runtime: RuntimeConfig
    loader: LoaderConfig
    query: QueryConfig
    observability: ObservabilityConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_config() -> ContextManagerConfig:
    project_root = _project_root()

    # This file is intentionally small: only runtime/tuning values that may
    # differ across environments or baselines should live here.
    # The handwritten SQL files are still the source of truth for shape-level
    # values. If embedding dim ever changes in `scripts/embeddings.py`, manually
    # sync the matching `vector(24)` declarations in `sql/01_init.sql` for
    # `metadata_chunks.embedding` and in `sql/02_retrieval.sql` for
    # `hybrid_search(query_embedding vector(...))`.
    return ContextManagerConfig(
        runtime=RuntimeConfig(
            database_url=os.environ.get(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5433/context_demo",
            ),
            default_resource_owner=os.environ.get(
                "RESOURCE_OWNER",
                "demo",
            ),
            default_resource_namespace=os.environ.get(
                "RESOURCE_NAMESPACE",
                os.environ.get("CATALOG_NAMESPACE", "default"),
            ),
        ),
        loader=LoaderConfig(
            value_cardinality_limit=int(
                os.environ.get("VALUE_CARDINALITY_LIMIT", "25")
            ),
            data_dir=Path(os.environ.get("DATA_DIR", project_root / "data")).expanduser(),
            source_adapter=os.environ.get("SOURCE_ADAPTER", "demo_json"),
        ),
        query=QueryConfig(
            hybrid_search_limit=int(os.environ.get("HYBRID_SEARCH_LIMIT", "24")),
            top_candidate_columns=int(
                os.environ.get("TOP_CANDIDATE_COLUMNS", "5")
            ),
            chunk_type_weights=MappingProxyType(
                {
                    "column_definition": float(
                        os.environ.get("WEIGHT_COLUMN_DEFINITION", "1.2")
                    ),
                    "value_definition": float(
                        os.environ.get("WEIGHT_VALUE_DEFINITION", "1.0")
                    ),
                    "rule": float(os.environ.get("WEIGHT_RULE", "0.6")),
                }
            ),
        ),
        observability=ObservabilityConfig(
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            log_dir=Path(os.environ.get("LOG_DIR", project_root / "logs")),
        ),
    )
