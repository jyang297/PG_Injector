from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg

from context_manager_config import get_config
from query import (
    build_candidate_columns,
    build_prompt_bundle,
    build_query_embedding,
    build_retrieval_inputs,
    fetch_hybrid_results,
)


CONFIG = get_config()
DATABASE_URL = CONFIG.runtime.database_url
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    owner = "test_owner"
    namespace = "test.table_identity"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "column_descriptions.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "description": "Account lifecycle status.",
                },
                {
                    "table_name": "contracts",
                    "column_name": "status",
                    "description": "Contract approval status.",
                },
            ],
        )
        write_json(
            root / "unique_values.json",
            [
                {
                    "table_name": "accounts",
                    "column_name": "status",
                    "values": [
                        {"raw_value": "active", "value_gloss": "Account is active."}
                    ],
                },
                {
                    "table_name": "contracts",
                    "column_name": "status",
                    "values": [
                        {
                            "raw_value": "approved",
                            "value_gloss": "Contract is approved.",
                        }
                    ],
                },
            ],
        )
        write_json(root / "rules.json", [])

        subprocess.run(
            [
                sys.executable,
                "scripts/load_demo.py",
                "--owner",
                owner,
                "--namespace",
                namespace,
                "--data-dir",
                str(root),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM column_catalog
                WHERE resource_owner = %s
                  AND resource_namespace = %s
                ORDER BY table_name, column_name
                """,
                (owner, namespace),
            )
            columns = cur.fetchall()
            cur.execute(
                """
                SELECT chunk_key
                FROM metadata_chunks
                WHERE resource_owner = %s
                  AND resource_namespace = %s
                  AND chunk_type = 'column_definition'
                ORDER BY chunk_key
                """,
                (owner, namespace),
            )
            chunk_keys = [row[0] for row in cur.fetchall()]

    expected_columns = [
        ("accounts", "status"),
        ("contracts", "status"),
    ]
    if columns != expected_columns:
        raise SystemExit(
            f"table-qualified columns did not persist correctly: {columns}"
        )
    expected_chunk_keys = [
        "column_definition::accounts::status",
        "column_definition::contracts::status",
    ]
    if chunk_keys != expected_chunk_keys:
        raise SystemExit(
            f"table-qualified chunk keys did not persist correctly: {chunk_keys}"
        )

    query_cases = [
        ("account lifecycle status", ("accounts", "status")),
        ("contract approval status", ("contracts", "status")),
    ]
    with psycopg.connect(DATABASE_URL) as conn:
        for query_text, expected_identity in query_cases:
            retrieval_inputs = build_retrieval_inputs(query_text)
            rows = fetch_hybrid_results(
                conn,
                owner,
                namespace,
                build_query_embedding(retrieval_inputs["semantic_query"]),
                retrieval_inputs["lexical_query"],
            )
            bundles = build_candidate_columns(
                conn,
                owner,
                namespace,
                rows,
                retrieval_inputs["keywords"],
                top_columns=2,
            )
            seen = [(bundle["table_name"], bundle["column_name"]) for bundle in bundles]
            if expected_identity not in seen:
                raise SystemExit(
                    f"query-time rollup missed {expected_identity}: seen={seen}"
                )
            prompt_metadata = build_prompt_bundle(
                owner,
                namespace,
                query_text,
                rows,
                bundles,
                retrieval_inputs,
            )
            prompt_seen = [
                (bundle["table_name"], bundle["column_name"])
                for bundle in prompt_metadata["candidate_columns"]
            ]
            if expected_identity not in prompt_seen:
                raise SystemExit(
                    f"prompt bundle missed {expected_identity}: seen={prompt_seen}"
                )

    print("table identity passed")


if __name__ == "__main__":
    main()
