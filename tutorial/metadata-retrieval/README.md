# Metadata Retrieval Tutorial

This tutorial focuses on one concrete capability:

- the user asks a noisy question
- the system should retrieve the most relevant columns and values
- the final output should be a column-level metadata bundle for text-to-SQL

Example target behavior:

- query: `blocked by legal or waiting for exec approval`
- expected columns:
  - `compliance_posture`
  - `contract_state`

## What you will implement

You will implement the feature in the real repo files, not in this tutorial folder.

Target files:

- `scripts/normalization.py`
- `scripts/load_metadata.py`
- `scripts/query.py`
- `sql/01_init.sql`
- `sql/02_retrieval.sql`
- `scripts/test_metadata_recall.py`

This tutorial folder only contains comments, checklists, and starter signatures.

## Recommended order

1. Implement deterministic query normalization
2. Build `column_definition` chunks
3. Build `value_definition` chunks
4. Make the query path retrieve metadata and roll hits up to columns
5. Add the metadata recall regression
6. Reload the database and verify the query

## Suggested work loop

After each step, run:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/context_demo
python scripts/load_metadata.py
python scripts/query.py "blocked by legal or waiting for exec approval"
python scripts/test_metadata_recall.py
```

## Checkpoint questions

Step 1:

- Do `legal`, `privacy`, `compliance`, and related phrases normalize into a stable token family?

Step 2:

- Does every column now have a searchable `column_definition` chunk?

Step 3:

- Does every semantically strong value now have a searchable `value_definition` chunk?

Step 4:

- Does the query path rank columns, not just raw chunk hits?

Step 5:

- Does the regression fail when a key column disappears from the final prompt bundle?
