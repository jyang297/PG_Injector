from __future__ import annotations

import re


ALIAS_GROUPS = {
    # These canonical families are shared query-time anchors. Add terms here
    # only when they represent the same business concept across many phrasings.
    "approval_gate": {
        "approval",
        "approve",
        "approver",
        "approvals",
        "signoff",
        "signoffs",
        "sign",
    },
    "early_rollout": {"pilot", "trial", "preview", "beta"},
    "europe_region": {"eu", "europe", "gdpr"},
    "legal_review": {"legal", "privacy", "security", "compliance"},
    "live_rollout": {"ga", "production", "live"},
    "risk_warning": {"amber", "warning", "caution", "watch"},
    "roadmap_support": {"roadmap", "commitment", "backed", "committed"},
}

PHRASE_REPLACEMENTS = (
    # Phrase rewrites are intentionally lossy: they collapse noisy STT or
    # bilingual variants into the compact anchor tokens used by retrieval.
    (r"\bgeneral availability\b", " ga "),
    (r"\bgenerally available\b", " ga "),
    (r"\bearly access\b", " pilot "),
    (r"\blimited rollout\b", " pilot "),
    (r"\bwhite glove\b", " white_glove "),
    (r"\bhigh touch\b", " white_glove "),
    (r"\bstuck in legal\b", " blocked_by_legal legal blocked "),
    (r"\bwaiting on compliance\b", " blocked_by_legal compliance blocked "),
    (
        r"\bwaiting for exec(?:utive)? approval\b",
        " needs_exec_signoff executive approval ",
    ),
    (r"\bexec(?:utive)? sign ?off\b", " needs_exec_signoff executive signoff "),
    (r"\bunder review\b", " under_review review "),
    (r"\bnot blocked\b", " cleared "),
    (r"法务卡住|卡在法务|法律卡住|被法务卡住", " blocked_by_legal legal blocked "),
    (r"法务|法律", " legal "),
    (r"合规", " compliance "),
    (r"隐私", " privacy "),
    (r"安全审查|安全审核", " security review "),
    (r"审批|签字|签核", " approval signoff "),
    (r"高风险|严重风险", " high risk "),
    (r"预警|警告", " warning "),
    (r"试点|试用|灰度", " pilot "),
    (r"正式上线|已上线|生产可用", " ga production "),
    (r"暂停|停掉|停用", " paused "),
    (r"欧盟|欧洲", " eu "),
    (r"数据驻留|驻留", " residency "),
    (r"路线图", " roadmap "),
    (r"支持等级|支持层级", " support tier "),
)

TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")


def _prepare(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("/", " ")
    lowered = lowered.replace("-", " ")
    lowered = lowered.replace(".net", " dotnet ")
    lowered = lowered.replace("dot net", " dotnet ")
    lowered = lowered.replace("sign off", " signoff ")
    lowered = lowered.replace("sign-off", " signoff ")
    for pattern, replacement in PHRASE_REPLACEMENTS:
        lowered = re.sub(pattern, replacement, lowered)
    return lowered


def normalized_tokens(text: str) -> list[str]:
    # Deduping keeps prompt and retrieval text stable, which matters for both
    # instrumentation comparisons and deterministic toy embeddings.
    prepared = _prepare(text)
    tokens = TOKEN_RE.findall(prepared)

    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        for canonical, variants in ALIAS_GROUPS.items():
            if token in variants:
                expanded.append(canonical)

    seen = set()
    deduped = []
    for token in expanded:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def normalize_for_search(text: str) -> str:
    return " ".join(normalized_tokens(text))


def dedupe_terms(terms: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped
