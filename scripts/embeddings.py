from __future__ import annotations

import hashlib
import math
from collections import Counter

from normalization import normalized_tokens


EMBEDDING_DIM = 24

CANONICAL_ALIASES = {
    # This toy embedding layer intentionally injects a little domain semantics
    # so the repo remains runnable without an external embedding provider.
    "approved": "cleared_state",
    "approve": "approval_gate",
    "approval": "approval_gate",
    "awaiting": "pending_state",
    "blocked": "blocked",
    "stuck": "blocked",
    "halted": "blocked",
    "paused": "blocked",
    "cleared": "cleared_state",
    "concierge": "white_glove_support",
    "deprioritized": "frozen_state",
    "elevated": "premium_support",
    "eu_only": "europe_region",
    "exec": "approval_gate",
    "executive": "approval_gate",
    "frozen": "frozen_state",
    "ga": "live_rollout",
    "gdpr": "europe_region",
    "glove": "white_glove_support",
    "grandfathered": "legacy_pricing",
    "healthy": "stable_state",
    "hold": "blocked",
    "internal": "internal_only_state",
    "live": "live_rollout",
    "warning": "risk_warning",
    "caution": "risk_warning",
    "amber": "risk_warning",
    "red": "critical_risk",
    "critical": "critical_risk",
    "severe": "critical_risk",
    "signoff": "approval_gate",
    "legal": "legal_review",
    "privacy": "legal_review",
    "security": "legal_review",
    "compliance": "legal_review",
    "eu": "europe_region",
    "europe": "europe_region",
    "pilot": "early_rollout",
    "trial": "early_rollout",
    "exploratory": "not_committed",
    "legacy": "legacy_pricing",
    "metered": "usage_pricing",
    "paused": "frozen_state",
    "pending": "pending_state",
    "premium": "premium_support",
    "production": "live_rollout",
    "tentative": "not_committed",
    "committed": "roadmap_backed",
    "backed": "roadmap_backed",
    "renewal": "contract_renewal",
    "residency": "data_region",
    "review": "pending_state",
    "roadmap": "roadmap_backed",
    "seat": "seat_pricing",
    "stable": "stable_state",
    "usage": "usage_pricing",
    "white": "white_glove_support",
}

def _expanded_tokens(text: str) -> list[str]:
    # The toy embedder gets most of its semantic lift from these expansions, so
    # changes here can materially affect retrieval rankings.
    tokens = normalized_tokens(text)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        canonical = CANONICAL_ALIASES.get(token)
        if canonical:
            expanded.append(canonical)
    return expanded


def _unit_vector_for_token(token: str, dim: int) -> list[float]:
    values: list[float] = []
    for i in range(dim):
        digest = hashlib.sha256(f"{token}:{i}".encode("utf-8")).digest()
        raw = int.from_bytes(digest[:4], "big", signed=False)
        values.append((raw / 2147483648.0) - 1.0)

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0] * dim
    return [value / norm for value in values]


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    token_counts = Counter(_expanded_tokens(text))
    if not token_counts:
        return [0.0] * dim

    vector = [0.0] * dim
    for token, weight in token_counts.items():
        token_vector = _unit_vector_for_token(token, dim)
        for i in range(dim):
            vector[i] += token_vector[i] * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * dim
    return [value / norm for value in vector]


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"
