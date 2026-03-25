# Code Review

Review date: 2026-03-24

## Summary

No new blocking correctness issues were found in the current metadata retrieval path.

The core contracts now look coherent:

- `table_name` is required at the source boundary
- stable column identity is structured as `resource_owner + resource_namespace + table_name + column_name`
- `table_name::column_name` is generated only for logs and debugging
- the reserved `::` separator is now treated as invalid inside raw source identifiers
- namespace replacement is scoped and transactional
- prompt/debug split is real, not only documented
- high-cardinality columns still inject descriptions into prompt metadata even when their values are not chunked

## Residual risks

### 1. Backward-compatible value source fallback still exists

The adapter still falls back from `unique_values.json` to `value_catalog.json` if the primary file is missing.

Relevant code:

- `scripts/source_adapters.py:40-49`
- `scripts/load_metadata.py`

Why it matters:

- this is useful for backward compatibility
- but it is still a source-contract escape hatch
- future production hardening may want an explicit switch for whether fallback is allowed

Current assessment:

- acceptable for the POC
- not a blocker for current metadata retrieval work


### 2. Rule path is structurally present but not yet prompt-complete

The query path currently exposes matched rules via `rule_id`, `text_semantic`, and `candidate_columns`, but it does not yet inject a packed `rule_text` payload.

Relevant code:

- `scripts/query.py:341-403`

Why it matters:

- this is consistent with the current roadmap, where packed meta rule injection is still deferred
- but it means rule-RAG effectiveness cannot yet be evaluated end to end inside this repo

Current assessment:

- intentional gap
- should stay documented until rule packets are actually wired through

## Review conclusion

The repo is in a good state for the current phase:

- metadata ingestion
- table-qualified retrieval
- prompt metadata assembly
- instrumentation and regression testing

The next meaningful architecture step is not another retrieval refactor. It is to decide when to start exercising real packed meta rules against an external SQL-generation layer.
