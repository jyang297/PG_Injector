from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg

from context_manager_config import get_config
from metadata_catalog import log_column_identity
from query import build_query_embedding, build_retrieval_inputs, fetch_hybrid_results


CONFIG = get_config()
DATABASE_URL = CONFIG.runtime.database_url
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_namespace(owner: str, namespace: str, data_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/load_demo.py",
            "--owner",
            owner,
            "--namespace",
            namespace,
            "--data-dir",
            str(data_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def main():
    owner = "test_owner"
    owner_a = "test_owner_a"
    owner_b = "test_owner_b"
    namespace_a = "test.namespace_a"
    namespace_b = "test.namespace_b"
    shared_namespace = "test.shared_namespace"
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        root_a = Path(tmp_a)
        root_b = Path(tmp_b)
        write_json(
            root_a / "column_descriptions.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "description": "Account lifecycle status for customer accounts.",
                }
            ],
        )
        write_json(
            root_a / "unique_values.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "values": [
                        {"raw_value": "active", "value_gloss": "Account is active."}
                    ],
                }
            ],
        )
        write_json(root_a / "rules.json", [])

        write_json(
            root_b / "column_descriptions.json",
            [
                {
                    "table_name": "contracts",
                    "column_name": "approval_state",
                    "description": "Legal approval state for enterprise contracts.",
                }
            ],
        )
        write_json(
            root_b / "unique_values.json",
            [
                {
                    "table_name": "contracts",
                    "column_name": "approval_state",
                    "values": [
                        {
                            "raw_value": "blocked",
                            "value_gloss": "Legal approval is blocked.",
                        }
                    ],
                }
            ],
        )
        write_json(root_b / "rules.json", [])

        load_namespace(owner, namespace_a, root_a)
        load_namespace(owner, namespace_b, root_b)
        load_namespace(owner_a, shared_namespace, root_a)
        load_namespace(owner_b, shared_namespace, root_b)

        query_a = build_retrieval_inputs("account lifecycle status")
        query_b = build_retrieval_inputs("legal approval status")
        embedding_a = build_query_embedding(query_a["semantic_query"])
        embedding_b = build_query_embedding(query_b["semantic_query"])

        with psycopg.connect(DATABASE_URL) as conn:
            rows_a = fetch_hybrid_results(
                conn,
                owner,
                namespace_a,
                embedding_a,
                query_a["lexical_query"],
            )
            rows_b = fetch_hybrid_results(
                conn,
                owner,
                namespace_b,
                embedding_b,
                query_b["lexical_query"],
            )
            rows_owner_a = fetch_hybrid_results(
                conn,
                owner_a,
                shared_namespace,
                embedding_a,
                query_a["lexical_query"],
            )
            rows_owner_b = fetch_hybrid_results(
                conn,
                owner_b,
                shared_namespace,
                embedding_b,
                query_b["lexical_query"],
            )

    if not rows_a or not rows_b:
        raise SystemExit("namespace isolation test expected hits in both namespaces")
    if any(row["resource_owner"] != owner for row in rows_a):
        raise SystemExit("namespace_a query leaked rows from another owner")
    if any(row["resource_namespace"] != namespace_a for row in rows_a):
        raise SystemExit("namespace_a query leaked rows from another namespace")
    if any(row["resource_owner"] != owner for row in rows_b):
        raise SystemExit("namespace_b query leaked rows from another owner")
    if any(row["resource_namespace"] != namespace_b for row in rows_b):
        raise SystemExit("namespace_b query leaked rows from another namespace")
    top_a = log_column_identity(rows_a[0]["table_name"], rows_a[0]["column_name"])
    top_b = log_column_identity(rows_b[0]["table_name"], rows_b[0]["column_name"])
    if top_a != "accounts::status":
        raise SystemExit(f"namespace_a returned unexpected top column: {top_a}")
    if top_b != "contracts::approval_state":
        raise SystemExit(f"namespace_b returned unexpected top column: {top_b}")
    if top_a == top_b:
        raise SystemExit("namespace isolation test expected different catalog hits")
    if not rows_owner_a or not rows_owner_b:
        raise SystemExit("owner isolation test expected hits in both owners")
    if any(row["resource_owner"] != owner_a for row in rows_owner_a):
        raise SystemExit("owner_a query leaked rows from another owner")
    if any(row["resource_namespace"] != shared_namespace for row in rows_owner_a):
        raise SystemExit("owner_a query leaked rows from another namespace")
    if any(row["resource_owner"] != owner_b for row in rows_owner_b):
        raise SystemExit("owner_b query leaked rows from another owner")
    if any(row["resource_namespace"] != shared_namespace for row in rows_owner_b):
        raise SystemExit("owner_b query leaked rows from another namespace")
    top_owner_a = log_column_identity(
        rows_owner_a[0]["table_name"],
        rows_owner_a[0]["column_name"],
    )
    top_owner_b = log_column_identity(
        rows_owner_b[0]["table_name"],
        rows_owner_b[0]["column_name"],
    )
    if top_owner_a != "accounts::status":
        raise SystemExit(f"owner_a returned unexpected top column: {top_owner_a}")
    if top_owner_b != "contracts::approval_state":
        raise SystemExit(f"owner_b returned unexpected top column: {top_owner_b}")
    if top_owner_a == top_owner_b:
        raise SystemExit("owner isolation test expected different catalog hits")

    print("namespace isolation passed")
