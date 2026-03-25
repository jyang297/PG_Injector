# SQL Checklist

Target files: `sql/01_init.sql`, `sql/02_retrieval.sql`

## What to change

### 1. Create metadata source tables

You should have:

- `column_catalog`
- `column_value_catalog`
- `rule_catalog`

### 2. Create one unified retrieval table

The retrieval table should be `metadata_chunks`.

It should allow these chunk types:

- `column_definition`
- `value_definition`
- `rule`

### 3. Index both lexical and vector views

You need:

- a `GIN` index on `search_tsv`
- an `HNSW` index on `embedding`

### 4. Keep the hybrid search SQL generic

The main fusion logic can stay generic and live in `sql/02_retrieval.sql`.

What matters is that:

- the metadata chunk exists
- it is searchable
- the query path can roll the result back up to columns

### 5. Rebuild the DB when needed

If you changed the init SQL, remember that the init script only runs on a fresh volume.

Reset flow:

```bash
docker compose down -v
docker compose up -d
python scripts/load_metadata.py
```
