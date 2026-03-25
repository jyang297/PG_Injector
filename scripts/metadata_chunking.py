from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from metadata_catalog import ColumnRef, MetadataCatalog, ValueSpec, log_column_identity
from normalization import dedupe_terms, normalized_tokens
from utils.text import compact_sentence

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
    "status",
    "such",
    "that",
    "the",
    "this",
    "tells",
    "to",
    "whether",
    "with",
}


@dataclass(frozen=True)
class ChunkProjection:
    chunks: list[dict[str, Any]]
    value_cardinality_by_column: dict[ColumnRef, int]
    mandatory_columns: set[ColumnRef]
    value_indexing: dict[str, Any]


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


def group_values_by_column(values: list[ValueSpec]) -> dict[ColumnRef, list[ValueSpec]]:
    grouped: dict[ColumnRef, list[ValueSpec]] = defaultdict(list)
    for value in values:
        grouped[value.ref].append(value)
    return dict(grouped)


def should_index_values(
    values: list[ValueSpec],
    *,
    value_cardinality_limit: int,
) -> bool:
    return len(values) <= value_cardinality_limit


def build_chunk_projection(
    catalog: MetadataCatalog,
    *,
    value_cardinality_limit: int,
) -> ChunkProjection:
    # Retrieval works on metadata chunks rather than source rows, so this
    # module owns the canonical projection from the internal catalog into
    # retrieval text plus stable prompt-injection policy flags.
    column_map = {column.ref: column for column in catalog.columns}
    value_groups = group_values_by_column(catalog.values)
    value_cardinality_by_column = {
        column_ref: len(values)
        for column_ref, values in value_groups.items()
    }
    mandatory_columns = {
        column_ref
        for column_ref, values in value_groups.items()
        if values
        and not should_index_values(
            values,
            value_cardinality_limit=value_cardinality_limit,
        )
    }

    chunks = []
    indexed_value_groups = 0
    indexed_value_count = 0

    for column in catalog.columns:
        table_name = column.table_name
        column_name = column.column_name
        raw_column_name = column.raw_column_name
        aliases = dedupe_terms(column.aliases)
        description = column.description

        text_exact = build_text_exact(
            table_name,
            column_name,
            aliases,
            description_keywords(description),
        )
        text_semantic = compact_sentence(description)
        payload = {
            "resource_owner": catalog.resource_owner,
            "resource_namespace": catalog.resource_namespace,
            "table_name": table_name,
            "column_name": column_name,
            "raw_column_name": raw_column_name,
            "description": description,
            "aliases": aliases,
            "data_type": column.data_type,
            "value_cardinality": value_cardinality_by_column.get(column.ref, 0),
            "mandatory_description_in_prompt": column.ref in mandatory_columns,
        }
        chunks.append(
            {
                "resource_owner": catalog.resource_owner,
                "resource_namespace": catalog.resource_namespace,
                "chunk_key": f"column_definition::{table_name}::{column_name}",
                "chunk_type": "column_definition",
                "table_name": table_name,
                "column_name": column_name,
                "rule_id": None,
                "raw_value": None,
                "text_exact": text_exact,
                "text_semantic": text_semantic,
                "payload": payload,
            }
        )

    for column_ref, values in value_groups.items():
        column = column_map.get(column_ref)
        if column is None:
            continue

        # Only semantically strong, low-cardinality values become retrieval
        # chunks. High-cardinality fields would add noise and inflate prompts.
        if not should_index_values(
            values,
            value_cardinality_limit=value_cardinality_limit,
        ):
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
                "resource_owner": catalog.resource_owner,
                "resource_namespace": catalog.resource_namespace,
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
                    "resource_owner": catalog.resource_owner,
                    "resource_namespace": catalog.resource_namespace,
                    "chunk_key": (
                        f"value_definition::{table_name}::{column_name}::{raw_value}"
                    ),
                    "chunk_type": "value_definition",
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
        candidate_columns = [
            {
                "table_name": ref.table_name,
                "column_name": ref.column_name,
            }
            for ref in rule.candidate_columns
        ]
        candidate_labels = [ref.log_label for ref in rule.candidate_columns]
        trigger_terms = dedupe_terms(rule.trigger_terms)
        description = rule.description
        text_exact = build_text_exact(rule_id, trigger_terms, candidate_labels)
        text_semantic = compact_sentence(description)
        payload = {
            "resource_owner": catalog.resource_owner,
            "resource_namespace": catalog.resource_namespace,
            "candidate_columns": candidate_columns,
            "intent": rule.intent,
            "priority": int(rule.priority),
            "trigger_terms": trigger_terms,
            "rule_text": rule.rule_text,
        }
        chunks.append(
            {
                "resource_owner": catalog.resource_owner,
                "resource_namespace": catalog.resource_namespace,
                "chunk_key": f"rule::{rule_id}",
                "chunk_type": "rule",
                "table_name": None,
                "column_name": None,
                "rule_id": rule_id,
                "raw_value": None,
                "text_exact": text_exact,
                "text_semantic": text_semantic,
                "payload": payload,
            }
        )

    return ChunkProjection(
        chunks=chunks,
        value_cardinality_by_column=value_cardinality_by_column,
        mandatory_columns=mandatory_columns,
        value_indexing={
            "indexed_value_groups": indexed_value_groups,
            "indexed_values": indexed_value_count,
            "total_value_groups": len(value_groups),
            "total_values": len(catalog.values),
            "mandatory_description_columns": len(mandatory_columns),
            "mandatory_description_labels": sorted(
                log_column_identity(ref.table_name, ref.column_name)
                for ref in mandatory_columns
            ),
        },
    )
