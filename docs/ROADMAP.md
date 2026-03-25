# Roadmap

## Current scope

The current implementation focuses on metadata retrieval for text-to-SQL:

- retrieve `column_definition` and `value_definition`
- roll chunk hits up to table-qualified columns
- assemble compact `prompt_metadata`
- keep schema and retrieval contracts stable while source adapters absorb raw JSON variability

The current repo boundary stops after metadata retrieval and prompt assembly.

- this repo does not generate SQL
- this repo does not call an LLM API for SQL generation
- this repo does not own the final SQL execution orchestration

See [DATAFLOW.md](./DATAFLOW.md) for the system boundary diagram.

Current source-contract note:

- raw source identifiers must not contain the reserved `::` separator because `table_name::column_name` is reserved for logs and debugging only

## Planned meta rule contract

Rule expansion is intentionally deferred. The next step is not a large per-column rule system.

The target shape is:

- rules stay as standalone metadata assets
- one matched rule is injected as one packed `meta rule`
- a packed `meta rule` may contain:
  - rewrite guidance
  - SQL-generation constraints
  - a small example block

This is preferred over attaching multiple rule groups to every column because:

- it is lighter to author and maintain
- it keeps rule evaluation independent from column metadata evaluation
- it allows clean ablation tests:
  - metadata only
  - metadata + rules
  - later, metadata + retrieved rule subsets

## Planned rule packet shape

The exact storage may change, but the intended contract is:

```json
{
  "rule_id": "enterprise_contract_gate",
  "trigger_terms": ["exec approval", "signoff", "legal block"],
  "candidate_columns": [
    {"table_name": "app_metadata", "column_name": "contract_state"},
    {"table_name": "app_metadata", "column_name": "compliance_posture"}
  ],
  "text_semantic": "Use this rule when the query is about commercial or legal blockers.",
  "rule_text": "Rewrite hints: ...\nSQL rules: ...\nExamples: ..."
}
```

Notes:

- `candidate_columns` is optional routing metadata, not a requirement that every rule be column-bound
- `rule_text` is the packed prompt-facing payload
- examples should stay small because rule payloads compete directly with retrieved metadata for prompt budget

## Rollout order

1. Keep the existing placeholder `rule` path and current schema.
2. Review the packed `meta rule` contract in docs first.
3. Add a small hand-authored `rules.json` with a few realistic rule packets.
4. Measure prompt impact and SQL quality with:
   - metadata only
   - metadata + packed rules
5. Only after that, decide whether rule retrieval needs finer-grained chunking or a separate retrieval path.

## Explicit non-goals for the POC

- no per-column triple rule bundles
- no separate `rewrite_rule` / `sql_rule` / `sql_example` schema yet
- no immediate schema refactor purely for rule experimentation

The assumption is that packed standalone rules are enough for the POC, and that later rule-RAG changes can be added without major architectural disruption.
