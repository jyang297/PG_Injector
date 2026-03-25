from __future__ import annotations

# Goal:
# Implement a small deterministic normalization layer for metadata retrieval.
#
# This is NOT meant to be a full NLP pipeline.
# It only needs enough logic to stabilize noisy user phrasing such as:
# - "blocked by legal"
# - "waiting for exec approval"
# - "欧盟数据驻留"
#
# After normalization, those surface forms should contribute to the same
# canonical search families.


ALIAS_GROUPS = {
    # TODO:
    # Add canonical token families such as:
    # - legal_review
    # - approval_gate
    # - europe_region
}


def _prepare(text: str) -> str:
    # TODO:
    # Lowercase the text.
    # Normalize obvious phrase variants such as:
    # - "waiting for exec approval"
    # - "stuck in legal"
    # - Chinese business phrases when useful
    #
    # Return the prepared text.
    raise NotImplementedError


def normalized_tokens(text: str) -> list[str]:
    # TODO:
    # 1. Call _prepare()
    # 2. Extract tokens
    # 3. Expand each token into its canonical alias family when needed
    # 4. Return a deduplicated token list
    raise NotImplementedError


def normalize_for_search(text: str) -> str:
    # TODO:
    # Join normalized_tokens() into one retrieval-friendly string.
    raise NotImplementedError


def dedupe_terms(terms: list[str]) -> list[str]:
    # TODO:
    # Deduplicate while preserving order.
    raise NotImplementedError
