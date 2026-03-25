# Metadata Retriever Demo

This demo shows how to combine PostgreSQL full-text search (`tsvector`) with `pgvector` for a metadata retrieval flow that supports text-to-SQL.

This repo stops at metadata retrieval and prompt assembly.

- it does not call an LLM API for SQL generation
- it does not generate SQL itself
- it does not own the final business-data execution pipeline

See [docs/DATAFLOW.md](/Users/codingleo/Code/Context_Manager/PG_test/docs/DATAFLOW.md) for the end-to-end boundary diagram.

## Why this shape

The target here is not row-level RAG over a business table.

The retrieval target is metadata:

- column definitions
- semantically strong values
- rules that will later constrain SQL generation

The LLM receives:

- normalized user input
- `column_definition`
- `value_definition`
- `rule` chunks when available

It then generates SQL, and PostgreSQL still executes the real query.

Instead of embedding data rows, the loader breaks metadata into retrieval-sized chunks stored in one unified `metadata_chunks` table.

The core retrieval contract is now owner + namespace aware:

- one database can hold multiple metadata catalogs
- each load replaces only one `resource_owner/resource_namespace`
- each query searches within one `resource_owner/resource_namespace`

Inside each resource scope, columns are now table-qualified:

- stable identity is `table_name + column_name`
- duplicate column names are allowed across different tables
- prompt metadata carries `table_name` and `column_name`
- `table_name::column_name` is now generated only for logs and human debugging

## What `tsvector` does here

`tsvector` is used for exact or near-exact recall:

- column names and raw names
- aliases and abbreviations
- raw values and synonyms
- compact business anchor terms

That is why the schema stores a `text_exact` field and indexes it with a weighted `GIN` index.

## What `pgvector` does here

`pgvector` is used for semantic-ish recall across:

- natural-language column meaning
- abstract business phrasing
- STT-style fuzzy wording
- rule semantics

This repo uses a deterministic toy embedding provider so the pipeline is runnable without an external model API. The provider is intentionally replaceable. In production, swap it for a real embedding model and keep the schema and query shape.

## Files

- `docker-compose.yml`: Postgres with `pgvector`
- `sql/01_init.sql`: extensions, source tables, `metadata_chunks`, and indexes
- `sql/02_retrieval.sql`: `hybrid_search()` retrieval function
- `data/column_descriptions.json`: column semantics
- `data/unique_values.json`: low-cardinality value metadata
- `data/rules.json`: reserved rule-RAG input, empty in V1
- `scripts/metadata_catalog.py`: internal canonical catalog types shared across loaders
- `scripts/metadata_chunking.py`: chunking policy and high-cardinality prompt-injection marking
- `scripts/source_adapters.py`: raw-source adapters that map JSON into the canonical catalog
- `scripts/embeddings.py`: toy embedding provider
- `scripts/load_metadata.py`: thin loader entrypoint that writes source tables and `metadata_chunks`
- `scripts/query.py`: runs hybrid retrieval, rolls hits up to columns, and prints an LLM-ready metadata bundle
- `scripts/test_source_adapter_validation.py`: validates adapter-side duplicate and rule-candidate handling
- `scripts/test_namespace_isolation.py`: checks namespace-scoped loading and retrieval isolation
- `scripts/test_table_identity.py`: checks that same-named columns from different tables persist as distinct identities
- `scripts/test_mandatory_column_injection.py`: checks that high-cardinality columns still inject descriptions into prompt metadata
- `scripts/context_manager_config.py`: shared runtime and tuning config for Python code paths
- `scripts/utils/observability.py`: reusable logging, timing, and payload-length instrumentation
- `docs/README.md`: docs index
- `docs/DATAFLOW.md`: metadata retriever dataflow and repo boundary
- `docs/ROADMAP.md`: planned evolution and packed meta rule direction
- `docs/CODE_REVIEW.md`: latest review summary and residual risks

## Repo boundary

This repository owns:

- source metadata ingestion
- canonical metadata normalization
- PostgreSQL metadata indexing
- hybrid retrieval
- column rollup
- `prompt_metadata` assembly

This repository does not own:

- SQL-generation prompts
- LLM API calling
- generated SQL validation
- final business-data query execution orchestration

Those steps should live in an external service or caller that consumes the retrieved `prompt_metadata`.

## Quick start

Start PostgreSQL:

```bash
docker compose up -d
```

Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Load the demo data:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/context_demo
export RESOURCE_OWNER=demo
export RESOURCE_NAMESPACE=default
python scripts/load_metadata.py
python scripts/load_metadata.py --owner demo --namespace shadow
```

Run a query:

```bash
python scripts/query.py --owner demo --namespace default "blocked by legal or waiting for exec approval"
python scripts/query.py --owner demo --namespace shadow "blocked by legal or waiting for exec approval"
python scripts/query.py --owner demo --namespace default "which column sounds like EU-only data hosting"
python scripts/query.py --owner demo --namespace default "still pilot but committed roadmap support"
python scripts/query.py --owner demo --namespace default "欧盟数据驻留和路线图承诺"
LOG_LEVEL=DEBUG python scripts/query.py --owner demo --namespace default "blocked by legal or waiting for exec approval"
python scripts/test_metadata_recall.py
python scripts/test_source_adapter_validation.py
python scripts/test_namespace_isolation.py
python scripts/test_table_identity.py
python scripts/test_mandatory_column_injection.py
```

Logs are written to `logs/load_metadata.log` and `logs/query.log` by default.
Set `LOG_LEVEL` or `LOG_DIR` to control verbosity and destination.

## Retrieval flow

1. Normalize the raw user query.
2. Build a canonical lexical query for `tsvector`.
3. Build an embedding query for `pgvector` from the original user phrasing so semantic recall still sees STT noise and abstract wording.
4. Run hybrid search over `metadata_chunks` within one `resource_owner/resource_namespace`.
5. Fuse lexical and semantic ranks.
6. Roll chunk hits up to the column level.
7. Force-inject column descriptions for high-cardinality columns whose raw values were intentionally not chunked.
8. Build an LLM-ready bundle:
   - normalized query
   - resource owner
   - resource namespace
   - table-qualified column identity via `table_name + column_name`
   - top candidate columns
   - matched values
   - matched rules
9. Return `prompt_metadata` to an external SQL-generation layer.

The SQL-generation LLM call is intentionally outside this repo.

## Instrumentation

The query and loader scripts both emit:

- stage-level timing
- namespace and adapter selection
- source-file selection and validation summaries
- retrieval and chunk-count summaries
- prompt payload size metrics
- estimated token counts for the final injected prompt bundle

When you run `python scripts/query.py "...query..." --json`, the output now includes:

- `candidate_columns_debug`
- `prompt_metadata`
- `instrumentation`

The instrumentation helpers live in `scripts/utils/observability.py` so they can be copied into another baseline or original solution with minimal changes.
The shared Python runtime/tuning defaults live in `scripts/context_manager_config.py`.
Schema-shaping constants are intentionally kept out of that file because the handwritten SQL in `sql/01_init.sql` and `sql/02_retrieval.sql` is still the source of truth.

## Adapter boundary

Raw source variability should stop at the adapter layer.

- `scripts/source_adapters.py` handles file names, nested JSON shapes, legacy `column_key` compatibility, and source-specific validation.
- `scripts/metadata_catalog.py` defines the internal catalog contract used by the loader.
- `scripts/load_metadata.py` only turns the canonical catalog into source-table rows and retrieval chunks.
- duplicate source keys are treated as fatal validation issues before any namespace replacement happens.
- `table_name` is part of the stable source contract; if raw metadata omits it, the adapter records validation issues and the loader fails before replacing the namespace.
- raw source identifiers must not contain the reserved `::` separator because `table_name::column_name` is reserved as a log/debug label.
- columns whose value groups are too large for `value_definition` chunking are marked for mandatory description injection so the SQL generator still sees those fields in prompt metadata.

This keeps the chunking, schema, retrieval, and prompt contracts stable even when source JSON formats change.

## Rule RAG readiness

V1 loads `column_definition` and `value_definition`.

The schema and loader already reserve:

- `rule_catalog`
- `chunk_type = rule`
- `matched_rules` in the prompt bundle

For the current POC, rule design stays intentionally simple:

- rules remain standalone assets rather than per-column rule bundles
- a matched rule can carry one packed `meta rule` payload for prompt injection
- that packed payload may include rewrite hints, SQL constraints, and small examples together
- the contract will be documented and iterated before any schema or loader expansion

See [docs/ROADMAP.md](/Users/codingleo/Code/Context_Manager/PG_test/docs/ROADMAP.md) for the planned rule contract and rollout order.

## Important production notes

- Replace the toy embedding provider with a real multilingual model.
- If your queries are mostly Chinese, do not rely on PostgreSQL built-in FTS as the primary recall path.
- Keep long descriptions in `payload`, not in `text_exact`.
- Only index low-cardinality, semantically strong values into `value_definition`.
- Keep both `resource_owner` and `resource_namespace` explicit if you load more than one datasource.
- If one namespace contains multiple business tables, require explicit `table_name` in source metadata and qualified rule candidate columns in rules.
- Do not allow raw source identifiers to contain `::`; that delimiter is reserved for log/debug labels.
- Add auth or tenancy routing on top of `resource_owner/resource_namespace` before productionizing hybrid search.
- Tune `hnsw.ef_search` after you measure recall and latency.

## Resetting the demo

The schema init script only runs when PostgreSQL initializes the volume for the first time. If you need a clean reset:

```bash
docker compose down -v
docker compose up -d
```
