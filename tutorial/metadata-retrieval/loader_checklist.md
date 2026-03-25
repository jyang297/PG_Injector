# Loader Checklist

Target file: `scripts/load_demo.py`

## What to change

### 1. Normalize source metadata

- Accept `column_name` or `column_key`
- Accept `value_gloss` or `value_description`
- Preserve optional fields such as aliases and data type

### 2. Build one `column_definition` chunk per column

Each chunk should capture:

- the canonical column name
- optional raw/source column name
- aliases
- the column's main business meaning

This chunk should have:

- `chunk_type = "column_definition"`
- `column_name` filled
- `raw_value = None`
- `rule_id = None`

### 3. Build one `value_definition` chunk per `(column, value)`

Only do this for low-cardinality, semantically strong values.

Each chunk should capture:

- the parent column
- the raw value
- synonyms
- a short value gloss

This chunk should have:

- `chunk_type = "value_definition"`
- `column_name` filled
- `raw_value` filled
- `rule_id = None`

### 4. Keep long descriptions in `payload`

Use:

- `text_exact` for lexical anchor terms
- `text_semantic` for a concise natural-language gloss
- `payload` for longer descriptions and auxiliary metadata

### 5. Reserve rule chunks

If `rules.json` is present, load one `rule` chunk per rule.

### 6. Reload and verify

After implementing the loader changes:

```bash
python scripts/load_demo.py
python scripts/query.py "blocked by legal or waiting for exec approval"
```

Expected direction:

- `value_definition::compliance_posture::blocked_by_legal` should be searchable
- `value_definition::contract_state::needs_exec_signoff` should be searchable
- the final metadata bundle should contain both related columns
