from __future__ import annotations

import argparse
import json

import psycopg
from context_manager_config import get_config
from embeddings import embed_text, vector_literal
from normalization import normalize_for_search, normalized_tokens
from utils import RunInstrumentation, compact_sentence, configure_logger, count_by_key

CONFIG = get_config()
DATABASE_URL = CONFIG.runtime.database_url


COLUMN_DETAIL_SQL = """
SELECT
  catalog_namespace,
  column_key,
  table_name,
  column_name,
  raw_column_name,
  description,
  aliases,
  data_type
FROM column_catalog
WHERE catalog_namespace = %s
  AND column_key = ANY(%s)
"""

NOISE_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "by",
    "column",
    "columns",
    "find",
    "for",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "metadata",
    "of",
    "on",
    "or",
    "show",
    "still",
    "that",
    "the",
    "their",
    "there",
    "these",
    "which",
    "with",
    "哪些",
    "哪个",
    "什么",
    "情况",
    "帮我",
    "看看",
    "一下",
    "有关",
    "相关",
    "和",
    "或",
    "还是",
    "字段",
    "列",
    "值",
}

CHUNK_TYPE_WEIGHTS = CONFIG.query.chunk_type_weights


def normalize_query_terms(query_text: str) -> list[str]:
    # Normalized tokens drive lexical recall, coverage scoring, and prompt-side
    # canonicalization. The semantic embedding path intentionally keeps the
    # original user phrasing instead of collapsing down to this token bag.
    tokens = normalized_tokens(normalize_for_search(query_text))
    filtered = []
    seen = set()
    for token in tokens:
        if token in NOISE_WORDS:
            continue
        if len(token) == 1 and token.isascii():
            continue
        if token not in seen:
            filtered.append(token)
            seen.add(token)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid retrieval over metadata chunks."
    )
    parser.add_argument("query_text", help="User query to retrieve metadata for.")
    parser.add_argument(
        "--namespace",
        default=CONFIG.runtime.default_catalog_namespace,
        help="Catalog namespace to search within.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text output.",
    )
    return parser.parse_args()


def build_fts_query(normalized_query: str) -> str:
    tokens = normalize_query_terms(normalized_query)
    if not tokens:
        return normalized_query
    return " OR ".join(tokens)


def extract_keywords(normalized_query: str) -> list[str]:
    keywords = normalize_query_terms(normalized_query)
    if keywords:
        return keywords
    return normalized_tokens(normalized_query)


def build_retrieval_inputs(query_text: str) -> dict:
    # `normalized_query` is the compact canonical form passed forward to prompt
    # assembly and lexical retrieval. `semantic_query` stays close to the raw
    # user phrasing so pgvector still sees STT noise, abstraction, and
    # underspecified wording instead of only canonical keywords.
    raw_normalized = normalize_for_search(query_text)
    cleaned_terms = normalize_query_terms(query_text)
    normalized_query = " ".join(cleaned_terms) if cleaned_terms else raw_normalized
    lexical_query = build_fts_query(normalized_query)
    semantic_query = " ".join(query_text.split()).strip() or raw_normalized
    return {
        "raw_query": query_text,
        "normalized_query": normalized_query,
        "retrieval_rewrite": None,
        "lexical_query": lexical_query,
        "semantic_query": semantic_query,
        "keywords": extract_keywords(normalized_query),
    }


def build_query_embedding(semantic_query_text: str) -> str:
    return vector_literal(embed_text(semantic_query_text))


def fetch_hybrid_results(
    conn: psycopg.Connection,
    catalog_namespace: str,
    query_embedding: str,
    lexical_query_text: str,
    limit: int | None = None,
):
    effective_limit = limit or CONFIG.query.hybrid_search_limit
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT *
            FROM hybrid_search(%s, %s, %s::vector, %s)
            """,
            (catalog_namespace, lexical_query_text, query_embedding, effective_limit),
        )
        return cur.fetchall()


def keyword_hits_for_row(row, keywords: list[str]) -> set[str]:
    payload = row["payload"] or {}
    # Coverage is only a confidence/tie-breaker signal. Restrict it to exact
    # anchors so verbose descriptions cannot overpower stronger hybrid hits.
    haystack = " ".join(
        [
            str(row.get("text_exact") or ""),
            str(row.get("column_name") or ""),
            str(row.get("raw_value") or ""),
            str(payload.get("raw_column_name") or ""),
            " ".join(payload.get("aliases") or []),
            " ".join(payload.get("synonyms") or []),
            " ".join(payload.get("business_tags") or []),
            " ".join(payload.get("candidate_columns") or []),
            " ".join(payload.get("trigger_terms") or []),
        ]
    ).lower()
    return {keyword for keyword in keywords if keyword in haystack}


def fetch_column_details(
    conn: psycopg.Connection,
    catalog_namespace: str,
    column_keys: list[str],
):
    if not column_keys:
        return {}
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(COLUMN_DETAIL_SQL, (catalog_namespace, column_keys))
        return {row["column_key"]: row for row in cur.fetchall()}


def build_candidate_columns(
    conn: psycopg.Connection,
    catalog_namespace: str,
    rows,
    keywords: list[str],
    top_columns: int | None = None,
):
    effective_top_columns = top_columns or CONFIG.query.top_candidate_columns
    candidates = {}

    def ensure_bucket(column_key: str):
        return candidates.setdefault(
            column_key,
            {
                "column_key": column_key,
                "score": 0.0,
                "coverage_terms": set(),
                "matched_values": {},
                "matched_rules": {},
                "supporting_chunks": [],
            },
        )

    for row in rows:
        payload = row["payload"] or {}
        if row["chunk_type"] == "rule":
            candidate_columns = payload.get("candidate_columns") or []
            if not candidate_columns:
                continue
            per_column_bonus = (
                row["score"]
                * CHUNK_TYPE_WEIGHTS["rule"]
                / max(len(candidate_columns), 1)
            )
            for column_key in candidate_columns:
                bucket = ensure_bucket(column_key)
                bucket["score"] += per_column_bonus
                bucket["coverage_terms"].update(keyword_hits_for_row(row, keywords))
                bucket["matched_rules"][row["rule_id"]] = {
                    "rule_id": row["rule_id"],
                    "score": per_column_bonus,
                    "text_semantic": row["text_semantic"],
                }
            continue

        column_key = row["column_key"] or payload.get("parent_column_key")
        if not column_key:
            continue

        bucket = ensure_bucket(column_key)
        bucket["score"] += row["score"] * CHUNK_TYPE_WEIGHTS.get(row["chunk_type"], 1.0)
        bucket["coverage_terms"].update(keyword_hits_for_row(row, keywords))
        bucket["supporting_chunks"].append(
            {
                "chunk_key": row["chunk_key"],
                "chunk_type": row["chunk_type"],
                "score": row["score"],
                "column_key": row["column_key"],
                "raw_value": row["raw_value"],
            }
        )

        if row["chunk_type"] == "value_definition" and row["raw_value"]:
            matched_value = bucket["matched_values"].setdefault(
                row["raw_value"],
                {
                    "raw_value": row["raw_value"],
                    "score": 0.0,
                    # Matched values are prompt-facing evidence, so keep the
                    # explanation to a single gloss sentence instead of the
                    # longer offline payload text.
                    "value_gloss": compact_sentence(payload.get("value_gloss") or ""),
                    "synonyms": payload.get("synonyms") or [],
                },
            )
            matched_value["score"] += row["score"]

    column_details = fetch_column_details(conn, catalog_namespace, list(candidates))
    # Chunks are only evidence. SQL generation consumes ranked columns, so
    # hybrid score stays primary and lexical keyword coverage is only a
    # tie-breaker for similarly scored candidates.
    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item["score"],
            -len(item["coverage_terms"]),
            item["column_key"],
        ),
    )[:effective_top_columns]

    bundles = []
    for candidate in ranked:
        detail = column_details.get(candidate["column_key"], {})
        resolved_table_name = detail.get("table_name")
        resolved_column_name = detail.get("column_name")
        if (
            (not resolved_table_name or not resolved_column_name)
            and "::" in candidate["column_key"]
        ):
            resolved_table_name, resolved_column_name = candidate[
                "column_key"
            ].split("::", maxsplit=1)
        matched_values = sorted(
            candidate["matched_values"].values(),
            key=lambda item: (-item["score"], item["raw_value"]),
        )
        matched_rules = sorted(
            candidate["matched_rules"].values(),
            key=lambda item: (-item["score"], item["rule_id"]),
        )
        bundles.append(
            {
                "column_key": candidate["column_key"],
                "table_name": resolved_table_name,
                "column_name": resolved_column_name,
                "raw_column_name": detail.get("raw_column_name"),
                "score": candidate["score"],
                "coverage_terms": sorted(candidate["coverage_terms"]),
                "short_description": compact_sentence(detail.get("description") or ""),
                "aliases": detail.get("aliases") or [],
                "data_type": detail.get("data_type"),
                "matched_values": matched_values[:4],
                "matched_rules": matched_rules[:4],
                "supporting_chunks": sorted(
                    candidate["supporting_chunks"],
                    key=lambda item: (-item["score"], item["chunk_key"]),
                )[:6],
            }
        )
    return bundles


def build_prompt_candidate_columns(candidate_columns) -> list[dict]:
    # The SQL generator only needs compact grounding. Scores, coverage terms,
    # and supporting chunks stay in the debug path instead of inflating prompts.
    return [
        {
            "column_key": bundle["column_key"],
            "table_name": bundle["table_name"],
            "column_name": bundle["column_name"],
            "raw_column_name": bundle["raw_column_name"],
            "short_description": bundle["short_description"],
            "aliases": bundle["aliases"],
            "data_type": bundle["data_type"],
            "matched_values": [
                {
                    "raw_value": value["raw_value"],
                    "value_gloss": value["value_gloss"],
                }
                for value in bundle["matched_values"]
            ],
            "matched_rules": [
                {
                    "rule_id": rule["rule_id"],
                    "text_semantic": rule["text_semantic"],
                }
                for rule in bundle["matched_rules"]
            ],
        }
        for bundle in candidate_columns
    ]


def build_prompt_bundle(
    catalog_namespace: str,
    query_text: str,
    rows,
    candidate_columns,
    retrieval_inputs: dict,
):
    # Keep the prompt bundle compact and provider-agnostic so the same metadata
    # shape can be reused across different SQL-generation baselines.
    seen_rules = set()
    matched_rules = []
    for row in rows:
        if row["chunk_type"] != "rule" or row["rule_id"] in seen_rules:
            continue
        seen_rules.add(row["rule_id"])
        matched_rules.append(
            {
                "rule_id": row["rule_id"],
                "text_semantic": row["text_semantic"],
                "candidate_columns": (row["payload"] or {}).get("candidate_columns")
                or [],
            }
        )

    return {
        "catalog_namespace": catalog_namespace,
        "raw_query": query_text,
        "normalized_query": retrieval_inputs["normalized_query"],
        "retrieval_rewrite": retrieval_inputs["retrieval_rewrite"],
        "candidate_columns": build_prompt_candidate_columns(candidate_columns),
        "matched_rules": matched_rules,
    }


def build_instrumentation_summary(rows, candidate_columns) -> dict:
    # These metrics stay intentionally simple so they can be compared across
    # retrieval variants without provider-specific parsing.
    return {
        "hit_count": len(rows),
        "chunk_type_counts": count_by_key(rows, "chunk_type"),
        "top_hit_chunk_keys": [row["chunk_key"] for row in rows[:5]],
        "top_hit_columns": [row["column_key"] for row in rows[:5]],
        "candidate_column_count": len(candidate_columns),
        "matched_value_count": sum(
            len(candidate["matched_values"]) for candidate in candidate_columns
        ),
        "matched_rule_count": sum(
            len(candidate["matched_rules"]) for candidate in candidate_columns
        ),
    }


def print_text(
    catalog_namespace: str,
    retrieval_inputs: dict,
    rows,
    candidate_columns,
    run_metrics: dict,
):
    print(f"Catalog namespace: {catalog_namespace}")
    print(f"Raw query: {retrieval_inputs['raw_query']}")
    print(f"Normalized query: {retrieval_inputs['normalized_query']}")
    print(f"Lexical query: {retrieval_inputs['lexical_query']}")
    print()
    print("Top hybrid hits:")
    for idx, row in enumerate(rows[:8], start=1):
        column_label = (
            row["column_key"]
            or (
                f"{(row['payload'] or {}).get('table_name')}::{row['column_name']}"
                if row["column_name"]
                else None
            )
        )
        print(
            f"{idx}. {row['chunk_type']} | score={row['score']:.5f} | "
            f"column={column_label} | value={row['raw_value']} | "
            f"semantic={row['text_semantic']}"
        )

    print()
    print("Candidate columns (debug view):")
    if not candidate_columns:
        print("No candidate columns were retrieved.")
        return

    for idx, bundle in enumerate(candidate_columns, start=1):
        print(
            f"{idx}. {bundle['table_name']}::{bundle['column_name']} | "
            f"aggregate_score={bundle['score']:.5f}"
        )
        print(f"   raw column name: {bundle['raw_column_name'] or '-'}")
        print(f"   keyword coverage: {', '.join(bundle['coverage_terms']) or '-'}")
        print(f"   meaning: {bundle['short_description']}")
        if bundle["matched_values"]:
            for value in bundle["matched_values"]:
                print(
                    f"   - matched value {value['raw_value']} "
                    f"(score={value['score']:.5f})"
                )
                print(f"     value meaning: {value['value_gloss']}")
        else:
            print("   no matched values")
        if bundle["matched_rules"]:
            for rule in bundle["matched_rules"]:
                print(f"   - rule {rule['rule_id']}: {rule['text_semantic']}")

    prompt_metrics = run_metrics["payloads"].get("prompt_metadata", {})
    print()
    print("Instrumentation:")
    print(
        f"  total_elapsed_ms={run_metrics.get('total_elapsed_ms')} | "
        f"hybrid_hits={run_metrics.get('retrieval_summary', {}).get('hit_count')} | "
        f"candidate_columns={run_metrics.get('retrieval_summary', {}).get('candidate_column_count')}"
    )
    print(
        f"  prompt_chars={prompt_metrics.get('chars')} | "
        f"prompt_utf8_bytes={prompt_metrics.get('utf8_bytes')} | "
        f"prompt_estimated_tokens={prompt_metrics.get('estimated_tokens')}"
    )
    print("  llm_prompt_contract=prompt_metadata")


def main():
    args = parse_args()
    catalog_namespace = args.namespace

    logger = configure_logger("query")
    instrumentation = RunInstrumentation("query", logger=logger)
    instrumentation.record(
        "runtime_config",
        {
            "database_url": DATABASE_URL,
            "catalog_namespace": catalog_namespace,
            "hybrid_search_limit": CONFIG.query.hybrid_search_limit,
            "top_candidate_columns": CONFIG.query.top_candidate_columns,
            "chunk_type_weights": dict(CHUNK_TYPE_WEIGHTS),
        },
    )
    query_text = args.query_text
    with instrumentation.stage("build_retrieval_inputs"):
        retrieval_inputs = build_retrieval_inputs(query_text)

    instrumentation.record_text("raw_query", retrieval_inputs["raw_query"])
    instrumentation.record_text(
        "normalized_query", retrieval_inputs["normalized_query"]
    )
    instrumentation.record_text("lexical_query", retrieval_inputs["lexical_query"])
    instrumentation.record_text("semantic_query", retrieval_inputs["semantic_query"])

    with instrumentation.stage("build_query_embedding"):
        query_embedding = build_query_embedding(retrieval_inputs["semantic_query"])

    with psycopg.connect(DATABASE_URL) as conn:
        with instrumentation.stage("fetch_hybrid_results"):
            # Retrieval limit is intentionally configurable because baseline
            # comparisons often sweep recall/latency tradeoffs here first.
            rows = fetch_hybrid_results(
                conn,
                catalog_namespace,
                query_embedding,
                retrieval_inputs["lexical_query"],
            )
        with instrumentation.stage("build_candidate_columns"):
            # Top-N columns is the final context budget boundary before prompt
            # assembly, so it also lives in config for controlled experiments.
            candidate_columns = build_candidate_columns(
                conn,
                catalog_namespace,
                rows,
                retrieval_inputs["keywords"],
            )

    instrumentation.record(
        "retrieval_summary",
        build_instrumentation_summary(rows, candidate_columns),
    )

    with instrumentation.stage("build_prompt_metadata"):
        prompt_metadata = build_prompt_bundle(
            catalog_namespace,
            query_text,
            rows,
            candidate_columns,
            retrieval_inputs,
        )

    instrumentation.record_payload("prompt_metadata", prompt_metadata)
    run_metrics = instrumentation.finalize()

    if args.json:
        print(
            json.dumps(
                {
                    "catalog_namespace": catalog_namespace,
                    **retrieval_inputs,
                    "hits": rows,
                    "candidate_columns_debug": candidate_columns,
                    "prompt_metadata": prompt_metadata,
                    "instrumentation": run_metrics,
                },
                indent=2,
                default=str,
            )
        )
        return

    print_text(
        catalog_namespace,
        retrieval_inputs,
        rows,
        candidate_columns,
        run_metrics,
    )


if __name__ == "__main__":
    main()
