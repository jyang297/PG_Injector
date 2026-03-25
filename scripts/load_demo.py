from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import psycopg

from context_manager_config import get_config
from embeddings import embed_text, vector_literal
from metadata_catalog import MetadataCatalog, ValueSpec
from normalization import dedupe_terms, normalized_tokens
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
    "duplicate_column_key",
    "duplicate_value_key",
    "duplicate_rule_id",
    "missing_column_table_name",
    "missing_value_group_table_name",
}

DESCRIPTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "app",
    "before",
    "can",
    "current",
    "field",
    "for",
    "from",
    "how",
    "in",
    "indicate",
    "is",
    "it",
    "may",
    "means",
    "of",
    "or",
    "such",
    "status",
    "that",
    "the",
    "this",
    "tells",
    "to",
    "whether",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load metadata into the PostgreSQL hybrid retrieval demo."
    )
    parser.add_argument(
        "--namespace",
        default=CONFIG.runtime.default_catalog_namespace,
        help="Target catalog namespace to replace.",
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


def description_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = normalized_tokens(text)
    keywords = []
    seen = set()
    for token in tokens:
        if token in DESCRIPTION_STOPWORDS:
            continue
        if len(token) <= 2 and token.isascii():
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= limit:
            break
    return keywords


def build_text_exact(*parts) -> str:
    exact_terms: list[str] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, list):
            for item in part:
                exact_terms.extend(normalized_tokens(str(item)))
            continue
        exact_terms.extend(normalized_tokens(str(part)))
    return " ".join(dedupe_terms(exact_terms))


def group_values_by_column(values: list[ValueSpec]) -> dict[str, list[ValueSpec]]:
    grouped: dict[str, list[ValueSpec]] = defaultdict(list)
    for value in values:
        grouped[value.column_key].append(value)
    return dict(grouped)


def should_index_values(values: list[ValueSpec]) -> bool:
    return len(values) <= VALUE_CARDINALITY_LIMIT


def build_metadata_chunks(catalog: MetadataCatalog):
    # Retrieval works on metadata chunks rather than source rows, so this
    # function is the canonical projection from the internal catalog model into
    # retrieval text plus structured payload. Raw-source shape differences
    # should already have been absorbed by the source adapter layer.
    column_map = {column.column_key: column for column in catalog.columns}
    value_groups = group_values_by_column(catalog.values)
    chunks = []
    indexed_value_groups = 0
    indexed_value_count = 0

    for column in catalog.columns:
        column_key = column.column_key
        table_name = column.table_name
        column_name = column.column_name
        raw_column_name = column.raw_column_name
        aliases = dedupe_terms(column.aliases)
        description = column.description

        text_exact = build_text_exact(
            table_name,
            column_name,
            raw_column_name,
            aliases,
            description_keywords(description),
        )
        text_semantic = compact_sentence(description)
        payload = {
            "catalog_namespace": catalog.namespace,
            "column_key": column_key,
            "table_name": table_name,
            "column_name": column_name,
            "raw_column_name": raw_column_name,
            "description": description,
            "aliases": aliases,
            "data_type": column.data_type,
        }
        chunks.append(
            {
                "catalog_namespace": catalog.namespace,
                "chunk_key": f"column_definition::{table_name}::{column_name}",
                "chunk_type": "column_definition",
                "column_key": column_key,
                "table_name": table_name,
                "column_name": column_name,
                "rule_id": None,
                "raw_value": None,
                "text_exact": text_exact,
                "text_semantic": text_semantic,
                "payload": payload,
            }
        )

    for column_key, values in value_groups.items():
        column = column_map.get(column_key)
        if column is None:
            continue

        # Only semantically strong, low-cardinality values become retrieval
        # chunks. High-cardinality fields would add noise and inflate prompts.
        if not should_index_values(values):
            continue

        indexed_value_groups += 1
        indexed_value_count += len(values)
        for value in values:
            table_name = value.table_name
            column_name = value.column_name
            raw_value = value.raw_value
            synonyms = dedupe_terms(value.synonyms)
            business_tags = dedupe_terms(value.business_tags)
            value_gloss = value.value_gloss

            text_exact = build_text_exact(
                table_name,
                column_name,
                raw_value,
                synonyms,
                business_tags,
                description_keywords(value_gloss, limit=6),
            )
            text_semantic = (
                f"For {table_name}.{column_name}, {raw_value} means "
                f"{compact_sentence(value_gloss)}"
            )
            payload = {
                "catalog_namespace": catalog.namespace,
                "parent_column_key": column_key,
                "table_name": table_name,
                "parent_column": column_name,
                "raw_column_name": column.raw_column_name,
                "raw_value": raw_value,
                "value_gloss": value_gloss,
                "synonyms": synonyms,
                "business_tags": business_tags,
            }
            chunks.append(
                {
                    "catalog_namespace": catalog.namespace,
                    "chunk_key": f"value_definition::{table_name}::{column_name}::{raw_value}",
                    "chunk_type": "value_definition",
                    "column_key": column_key,
                    "table_name": table_name,
                    "column_name": column_name,
                    "rule_id": None,
                    "raw_value": raw_value,
                    "text_exact": text_exact,
                    "text_semantic": text_semantic,
                    "payload": payload,
                }
            )

    for rule in catalog.rules:
        # Rule chunks share the same storage and retrieval path so later rule
        # RAG can plug into the existing hybrid search and column rollup flow.
        rule_id = rule.rule_id
        candidate_columns = dedupe_terms(rule.candidate_columns)
        trigger_terms = dedupe_terms(rule.trigger_terms)
        intent = rule.intent
        priority = int(rule.priority)
        description = rule.description
        text_exact = build_text_exact(rule_id, trigger_terms, candidate_columns)
        text_semantic = compact_sentence(description)
        payload = {
            "catalog_namespace": catalog.namespace,
            "candidate_columns": candidate_columns,
            "intent": intent,
            "priority": priority,
            "trigger_terms": trigger_terms,
            "rule_text": rule.rule_text,
        }
        chunks.append(
            {
                "catalog_namespace": catalog.namespace,
                "chunk_key": f"rule::{rule_id}",
                "chunk_type": "rule",
                "column_key": None,
                "table_name": None,
                "column_name": None,
                "rule_id": rule_id,
                "raw_value": None,
                "text_exact": text_exact,
                "text_semantic": text_semantic,
                "payload": payload,
            }
        )

    return chunks, {
        "indexed_value_groups": indexed_value_groups,
        "indexed_values": indexed_value_count,
        "total_value_groups": len(value_groups),
        "total_values": len(catalog.values),
    }


def replace_namespace(cur: psycopg.Cursor, namespace: str) -> None:
    # Namespace-scoped replacement keeps one datasource refresh from deleting
    # every other catalog in the same database.
    cur.execute(
        "DELETE FROM rule_catalog WHERE catalog_namespace = %s",
        (namespace,),
    )
    cur.execute(
        "DELETE FROM column_catalog WHERE catalog_namespace = %s",
        (namespace,),
    )


def fatal_validation_issues(catalog: MetadataCatalog):
    return [
        issue
        for issue in catalog.validation_issues
        if issue.issue_type in FATAL_VALIDATION_ISSUES
    ]


def main():
    args = parse_args()
    namespace = args.namespace
    data_dir = Path(args.data_dir).expanduser().resolve()
    source_adapter_name = args.source_adapter
    source_adapter = get_source_adapter(source_adapter_name)

    logger = configure_logger("load_demo")
    instrumentation = RunInstrumentation("load_demo", logger=logger)
    instrumentation.record(
        "runtime_config",
        {
            "database_url": DATABASE_URL,
            "catalog_namespace": namespace,
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
        catalog = source_adapter.load(data_dir, namespace)

    if not catalog.columns:
        raise SystemExit(
            f"Refusing to replace namespace {namespace!r} with an empty catalog."
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

    with instrumentation.stage("build_metadata_chunks"):
        chunks, value_indexing = build_metadata_chunks(catalog)

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
            f"{namespace!r} because fatal validation issues were found: "
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
                replace_namespace(cur, namespace)

            with instrumentation.stage(
                "db_insert_column_catalog",
                row_count=len(catalog.columns),
            ):
                # Structured source tables stay useful after chunking because the
                # prompt assembler can fetch clean column metadata from them.
                cur.executemany(
                    """
                    INSERT INTO column_catalog
                      (catalog_namespace, column_key, table_name, column_name,
                       raw_column_name, description, aliases, data_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            namespace,
                            column.column_key,
                            column.table_name,
                            column.column_name,
                            column.raw_column_name,
                            column.description,
                            column.aliases,
                            column.data_type,
                        )
                        for column in catalog.columns
                    ],
                )

            value_rows = [
                (
                    namespace,
                    value.column_key,
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
                      (catalog_namespace, column_key, raw_value,
                       value_description, synonyms, business_tags)
                    VALUES (%s, %s, %s, %s, %s, %s)
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
                          (catalog_namespace, rule_id, text_exact, text_semantic,
                           candidate_columns, intent, priority, payload)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        [
                            (
                                namespace,
                                rule.rule_id,
                                build_text_exact(
                                    rule.rule_id,
                                    rule.trigger_terms,
                                    rule.candidate_columns,
                                ),
                                compact_sentence(rule.description),
                                rule.candidate_columns,
                                rule.intent,
                                int(rule.priority),
                                json.dumps(
                                    {
                                        "catalog_namespace": namespace,
                                        "trigger_terms": rule.trigger_terms,
                                        "candidate_columns": rule.candidate_columns,
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
                            chunk["catalog_namespace"],
                            chunk["chunk_key"],
                            chunk["chunk_type"],
                            chunk["column_key"],
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
                      (catalog_namespace, chunk_key, chunk_type,
                       column_key, table_name, column_name, rule_id, raw_value,
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
                    WHERE catalog_namespace = %s
                    """,
                    (namespace,),
                )
                total_chunks = cur.fetchone()[0]

        with instrumentation.stage("db_commit"):
            conn.commit()

    run_metrics = instrumentation.finalize()

    print(f"Loaded demo data into {DATABASE_URL}")
    print(
        f"Namespace {namespace}: "
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
