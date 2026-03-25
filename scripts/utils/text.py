from __future__ import annotations

import re


def compact_sentence(text: str) -> str:
    # Prompt metadata should carry short glosses, not full prose blobs. Keeping
    # the shortening rule in one shared helper prevents loader/query drift.
    stripped = " ".join(str(text or "").split()).strip()
    if not stripped:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", stripped, maxsplit=1)[0]
    return first_sentence.rstrip(".") + "."
