from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import psycopg

from context_manager_config import get_config
from query import build_query_embedding, build_retrieval_inputs, fetch_hybrid_results


CONFIG = get_config()
DATABASE_URL = CONFIG.runtime.database_url
ROOT = Path(__file__).resolve().parents[1]


def load_namespace(namespace: str) -> None:
    subprocess.run(
        [sys.executable, "scripts/load_demo.py", "--namespace", namespace],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def main():
    namespace_a = "test.namespace_a"
    namespace_b = "test.namespace_b"
    load_namespace(namespace_a)
    load_namespace(namespace_b)

    query_text = "blocked by legal or waiting for exec approval"
    retrieval_inputs = build_retrieval_inputs(query_text)
    query_embedding = build_query_embedding(retrieval_inputs["semantic_query"])

    with psycopg.connect(DATABASE_URL) as conn:
        rows_a = fetch_hybrid_results(
            conn,
            namespace_a,
            query_embedding,
            retrieval_inputs["lexical_query"],
        )
        rows_b = fetch_hybrid_results(
            conn,
            namespace_b,
            query_embedding,
            retrieval_inputs["lexical_query"],
        )

    if not rows_a or not rows_b:
        raise SystemExit("namespace isolation test expected hits in both namespaces")
    if any(row["catalog_namespace"] != namespace_a for row in rows_a):
        raise SystemExit("namespace_a query leaked rows from another namespace")
    if any(row["catalog_namespace"] != namespace_b for row in rows_b):
        raise SystemExit("namespace_b query leaked rows from another namespace")

    print("namespace isolation passed")
