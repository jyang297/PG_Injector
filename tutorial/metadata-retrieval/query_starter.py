from __future__ import annotations

# Goal:
# Make the query path aware of metadata chunks.
#
# You are not building a full SQL planner here.
# You only need enough logic to:
# - normalize noisy queries
# - build a compact FTS query
# - retrieve metadata chunks
# - roll chunk hits up to columns


NOISE_WORDS = {
    # TODO:
    # Add the obvious filler words you do not want in the final lexical query.
}


def build_retrieval_inputs(query_text: str) -> dict:
    # TODO:
    # Return a dict containing at least:
    # - raw_query
    # - normalized_query
    # - lexical_query
    # - semantic_query
    # - keywords
    raise NotImplementedError


def keyword_hits_for_row(row, keywords: list[str]) -> set[str]:
    # TODO:
    # Build a haystack from metadata fields such as:
    # - column_name
    # - raw_value
    # - column description
    # - value gloss
    # - synonyms
    #
    # Return which query keywords this row covers.
    raise NotImplementedError


def build_candidate_columns(conn, rows, keywords: list[str], top_columns: int = 5):
    # TODO:
    # Roll raw chunk hits up to the column level.
    #
    # Rules of thumb:
    # - column_definition contributes directly to the column
    # - value_definition contributes to its parent column
    # - rule chunks can add a bonus to their candidate columns
    raise NotImplementedError
