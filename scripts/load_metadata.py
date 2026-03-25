from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import psycopg

from context_manager_config import get_config
from embeddings import embed_text, vector_literal
from metadata_catalog import MetadataCatalog
from metadata_chunking import build_chunk_projection, build_text_exact
from source_adapters import get_source_adapter
from utils import (
    RunInstrumentation,
    compact_sentence,
    configure_logger,
    count_by_key,
    summarize_text_collection,
)

CONFIG = get_config()
DATABASE_URL = CONFIG.runtime.database_url
VALUE_CARDINALITY_LIMIT = CONFIG.loader.value_cardinality_limit
FATAL_VALIDATION_ISSUES = {
    "duplicate_column_identity",
    "duplicate_value_key",
    "duplicate_rule_id",
    "invalid_columns_payload",
    "invalid_value_groups_payload",
    "invalid_rules_payload",
    "missing_column_table_name",
    "missing_value_group_table_name",
    "invalid_identifier_separator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load metadata into the PostgreSQL hybrid retrieval demo."
    )
    parser.add_argument(
        "--owner",
        default=CONFIG.runtime.default_resource_owner,
        help="Top-level resource owner to replace.",
    )
    parser.add_argument(
        "--namespace",
        default=CONFIG.runtime.default_resource_namespace,
        help="Target resource namespace to replace.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(CONFIG.loader.data_dir),
        help="Directory containing source metadata JSON files.",
    )
    parser.add_argument(
        "--source-adapter",
        default=CONFIG.loader.source_adapter,
        help="Source adapter used to map raw files into the internal catalog model.",
    )
    return parser.parse_args()


def replace_namespace(
    cur: psycopg.Cursor,
    resource_owner: str,
    resource_namespace: str,
) -> None:
    # Namespace-scoped replacement keeps one datasource refresh from deleting
    # every other catalog in the same database.
    cur.execute(
        """
        DELETE FROM rule_catalog
        WHERE resource_owner = %s
          AND resource_namespace = %s
        """,
        (resource_owner, resource_namespace),
    )
    cur.execute(
        """
        DELETE FROM column_catalog
        WHERE resource_owner = %s
          AND resource_namespace = %s
        """,
        (resource_owner, resource_namespace),
    )


def fatal_validation_issues(catalog: MetadataCatalog):
    return [
        issue
        for issue in catalog.validation_issues
        if issue.issue_type in FATAL_VALIDATION_ISSUES
    ]


def main():
    args = parse_args()
    resource_owner = args.owner
    resource_namespace = args.namespace
    data_dir = Path(args.data_dir).expanduser().resolve()
    source_adapter_name = args.source_adapter
    source_adapter = get_source_adapter(source_adapter_name)

    logger = configure_logger("load_metadata")
    instrumentation = RunInstrumentation("load_metadata", logger=logger)
    instrumentation.record(
        "runtime_config",
        {
            "database_url": DATABASE_URL,
            "resource_owner": resource_owner,
            "resource_namespace": resource_namespace,
            "data_dir": str(data_dir),
            "source_adapter": source_adapter_name,
            "table_name_contract": "required",
            "value_cardinality_limit": VALUE_CARDINALITY_LIMIT,
        },
    )

    with instrumentation.stage("load_source_files"):
        # The adapter boundary is where raw JSON variability is absorbed. After
        # this step the rest of the pipeline only deals with a stable internal
        # catalog contract.
        catalog = source_adapter.load(data_dir, resource_owner, resource_namespace)

    if not catalog.columns:
        raise SystemExit(
            "Refusing to replace "
            f"{resource_owner!r}/{resource_namespace!r} with an empty catalog."
        )

    value_catalog_source = catalog.source_files.get("values")
    if value_catalog_source and value_catalog_source != "unique_values.json":
        logger.warning(
            "value_catalog_fallback source={} expected={}",
            value_catalog_source,
            "unique_values.json",
        )

    instrumentation.record(
        "source_counts",
        {
            **catalog.source_counts,
            "value_cardinality_limit": VALUE_CARDINALITY_LIMIT,
        },
    )
    instrumentation.record(
        "catalog_sources",
        catalog.source_files,
    )

    with instrumentation.stage("build_chunk_projection"):
        chunk_projection = build_chunk_projection(
            catalog,
            value_cardinality_limit=VALUE_CARDINALITY_LIMIT,
        )
        chunks = chunk_projection.chunks
        value_indexing = chunk_projection.value_indexing

    instrumentation.record(
        "validation_summary",
        {
            "issue_count": len(catalog.validation_issues),
            "issue_type_counts": count_by_key(
                [asdict(issue) for issue in catalog.validation_issues],
                "issue_type",
            ),
            "sample_issues": [
                asdict(issue) for issue in catalog.validation_issues[:5]
            ],
        },
    )
    instrumentation.record("value_indexing", value_indexing)
    if catalog.validation_issues:
        logger.warning(
            "metadata_validation issues={} sample={}",
            len(catalog.validation_issues),
            [asdict(issue) for issue in catalog.validation_issues[:3]],
        )
    fatal_issues = fatal_validation_issues(catalog)
    if fatal_issues:
        raise SystemExit(
            "Refusing to load namespace "
            f"{resource_owner!r}/{resource_namespace!r} because fatal validation "
            f"issues were found: "
            f"{[asdict(issue) for issue in fatal_issues[:5]]}"
        )

    instrumentation.record(
        "chunk_summary",
        {
            "total_chunks": len(chunks),
            "chunk_type_counts": count_by_key(chunks, "chunk_type"),
            "text_exact": summarize_text_collection(
                [chunk["text_exact"] for chunk in chunks]
            ),
            "text_semantic": summarize_text_collection(
                [chunk["text_semantic"] for chunk in chunks]
            ),
        },
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            with instrumentation.stage("db_replace_namespace"):
                replace_namespace(cur, resource_owner, resource_namespace)

            with instrumentation.stage(
                "db_insert_column_catalog",
                row_count=len(catalog.columns),
            ):
                # Structured source tables stay useful after chunking because the
                # prompt assembler can fetch clean column metadata from them.
                cur.executemany(
                    """
                    INSERT INTO column_catalog
                      (resource_owner, resource_namespace, table_name, column_name,
                       raw_column_name, description, aliases, data_type,
                       value_cardinality, mandatory_description_in_prompt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            resource_owner,
                            resource_namespace,
                            column.table_name,
                            column.column_name,
                            column.raw_column_name,
                            column.description,
                            column.aliases,
                            column.data_type,
                            chunk_projection.value_cardinality_by_column.get(
                                column.ref,
                                0,
                            ),
                            column.ref in chunk_projection.mandatory_columns,
                        )
                        for column in catalog.columns
                    ],
                )

            value_rows = [
                (
                    resource_owner,
                    resource_namespace,
                    value.table_name,
                    value.column_name,
                    value.raw_value,
                    value.value_gloss,
                    value.synonyms,
                    value.business_tags,
                )
                for value in catalog.values
            ]
            with instrumentation.stage(
                "db_insert_value_catalog",
                row_count=len(value_rows),
            ):
                cur.executemany(
                    """
                    INSERT INTO column_value_catalog
                      (resource_owner, resource_namespace, table_name, column_name,
                       raw_value,
                       value_description, synonyms, business_tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    value_rows,
                )

            if catalog.rules:
                with instrumentation.stage(
                    "db_insert_rule_catalog",
                    row_count=len(catalog.rules),
                ):
                    cur.executemany(
                        """
                        INSERT INTO rule_catalog
                          (resource_owner, resource_namespace, rule_id,
                           text_exact, text_semantic,
                           candidate_columns, intent, priority, payload)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                        """,
                        [
                            (
                                resource_owner,
                                resource_namespace,
                                rule.rule_id,
                                build_text_exact(
                                    rule.rule_id,
                                    rule.trigger_terms,
                                    [ref.log_label for ref in rule.candidate_columns],
                                ),
                                compact_sentence(rule.description),
                                json.dumps(
                                    [
                                        {
                                            "table_name": ref.table_name,
                                            "column_name": ref.column_name,
                                        }
                                        for ref in rule.candidate_columns
                                    ]
                                ),
                                rule.intent,
                                int(rule.priority),
                                json.dumps(
                                    {
                                        "resource_owner": resource_owner,
                                        "resource_namespace": resource_namespace,
                                        "trigger_terms": rule.trigger_terms,
                                        "candidate_columns": [
                                            {
                                                "table_name": ref.table_name,
                                                "column_name": ref.column_name,
                                            }
                                            for ref in rule.candidate_columns
                                        ],
                                        "rule_text": rule.rule_text,
                                    }
                                ),
                            )
                            for rule in catalog.rules
                        ],
                    )

            chunk_rows = []
            with instrumentation.stage(
                "build_chunk_rows_with_embeddings",
                row_count=len(chunks),
            ):
                # Embeddings only use `text_semantic`. Exact anchor terms are
                # deliberately kept in `text_exact` for the lexical index.
                for chunk in chunks:
                    embedding = vector_literal(embed_text(chunk["text_semantic"]))
                    chunk_rows.append(
                        (
                            chunk["resource_owner"],
                            chunk["resource_namespace"],
                            chunk["chunk_key"],
                            chunk["chunk_type"],
                            chunk["table_name"],
                            chunk["column_name"],
                            chunk["rule_id"],
                            chunk["raw_value"],
                            chunk["text_exact"],
                            chunk["text_semantic"],
                            json.dumps(chunk["payload"]),
                            embedding,
                        )
                    )

            with instrumentation.stage(
                "db_insert_metadata_chunks",
                row_count=len(chunk_rows),
            ):
                cur.executemany(
                    """
                    INSERT INTO metadata_chunks
                      (resource_owner, resource_namespace, chunk_key, chunk_type,
                       table_name, column_name, rule_id, raw_value,
                       text_exact, text_semantic, payload, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    chunk_rows,
                )

            with instrumentation.stage("db_count_metadata_chunks"):
                cur.execute(
                    """
                    SELECT count(*)
                    FROM metadata_chunks
                    WHERE resource_owner = %s
                      AND resource_namespace = %s
                    """,
                    (resource_owner, resource_namespace),
                )
                total_chunks = cur.fetchone()[0]

        with instrumentation.stage("db_commit"):
            conn.commit()

    run_metrics = instrumentation.finalize()

    print(f"Loaded demo data into {DATABASE_URL}")
    print(
        f"Resource {resource_owner}/{resource_namespace}: "
        "Created "
        f"{len(catalog.columns)} columns, "
        f"{len(catalog.values)} values, "
        f"{len(catalog.rules)} rules, and {total_chunks} metadata chunks."
    )
    print(
        "Instrumentation: "
        f"total_elapsed_ms={run_metrics['total_elapsed_ms']} | "
        f"text_exact_tokens_total="
        f"{run_metrics['chunk_summary']['text_exact']['estimated_tokens_total']} | "
        f"text_semantic_tokens_total="
        f"{run_metrics['chunk_summary']['text_semantic']['estimated_tokens_total']}"
    )


if __name__ == "__main__":
    main()
