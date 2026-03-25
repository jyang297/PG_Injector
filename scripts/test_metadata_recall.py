from __future__ import annotations

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
CATALOG_NAMESPACE = CONFIG.runtime.default_catalog_namespace


def main():
    # Keep this regression small and deterministic. Its job is to catch obvious
    # retrieval regressions before running larger baseline comparisons.
    semantic_contract_query = "blocked by legal or waiting for exec approval"
    retrieval_contract = build_retrieval_inputs(semantic_contract_query)
    if retrieval_contract["semantic_query"] != semantic_contract_query:
        raise SystemExit(
            "semantic query should preserve original user phrasing for embeddings"
        )

    cases = [
        {
            "query": "blocked by legal or waiting for exec approval",
            "expected": {
                "app_metadata::compliance_posture",
                "app_metadata::contract_state",
            },
        },
        {
            "query": "欧盟数据驻留和路线图承诺",
            "expected": {
                "app_metadata::data_residency",
                "app_metadata::roadmap_commitment",
            },
        },
    ]

    with psycopg.connect(DATABASE_URL) as conn:
        for case in cases:
            retrieval_inputs = build_retrieval_inputs(case["query"])
            query_embedding = build_query_embedding(retrieval_inputs["semantic_query"])
            rows = fetch_hybrid_results(
                conn,
                CATALOG_NAMESPACE,
                query_embedding,
                retrieval_inputs["lexical_query"],
            )
            bundles = build_candidate_columns(
                conn,
                CATALOG_NAMESPACE,
                rows,
                retrieval_inputs["keywords"],
                top_columns=CONFIG.query.top_candidate_columns,
            )
            seen = {bundle["column_key"] for bundle in bundles}
            missing = sorted(case["expected"] - seen)
            if missing:
                raise SystemExit(
                    "metadata recall failed for "
                    f"{case['query']!r}; missing={missing}; seen={sorted(seen)}"
                )

            prompt_metadata = build_prompt_bundle(
                CATALOG_NAMESPACE,
                case["query"],
                rows,
                bundles,
                retrieval_inputs,
            )
            if not prompt_metadata["candidate_columns"]:
                raise SystemExit("prompt bundle should include candidate columns")
            forbidden_keys = {"score", "coverage_terms", "supporting_chunks"}
            leaked = forbidden_keys & set(prompt_metadata["candidate_columns"][0])
            if leaked:
                raise SystemExit(
                    f"prompt bundle leaked debug fields: {sorted(leaked)}"
                )
            candidate = prompt_metadata["candidate_columns"][0]
            required_keys = {"column_key", "table_name", "column_name"}
            missing_keys = required_keys - set(candidate)
            if missing_keys:
                raise SystemExit(
                    f"prompt bundle missing table-qualified identity fields: "
                    f"{sorted(missing_keys)}"
                )
            if "::" not in candidate["column_key"]:
                raise SystemExit(
                    "prompt bundle should expose table-qualified column_key values"
                )

    print("metadata recall passed")


if __name__ == "__main__":
    main()
