CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS column_catalog (
  catalog_namespace text NOT NULL,
  column_key text NOT NULL,
  table_name text NOT NULL,
  column_name text NOT NULL,
  raw_column_name text,
  description text NOT NULL,
  aliases text[] NOT NULL DEFAULT '{}',
  data_type text,
  PRIMARY KEY (catalog_namespace, column_key),
  UNIQUE (catalog_namespace, table_name, column_name)
);

CREATE TABLE IF NOT EXISTS column_value_catalog (
  catalog_namespace text NOT NULL,
  column_key text NOT NULL,
  raw_value text NOT NULL,
  value_description text NOT NULL,
  synonyms text[] NOT NULL DEFAULT '{}',
  business_tags text[] NOT NULL DEFAULT '{}',
  PRIMARY KEY (catalog_namespace, column_key, raw_value),
  FOREIGN KEY (catalog_namespace, column_key)
    REFERENCES column_catalog(catalog_namespace, column_key)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rule_catalog (
  catalog_namespace text NOT NULL,
  rule_id text NOT NULL,
  text_exact text NOT NULL,
  text_semantic text NOT NULL,
  candidate_columns text[] NOT NULL DEFAULT '{}',
  intent text,
  priority integer NOT NULL DEFAULT 0,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (catalog_namespace, rule_id)
);

CREATE TABLE IF NOT EXISTS metadata_chunks (
  id bigserial PRIMARY KEY,
  catalog_namespace text NOT NULL,
  chunk_key text NOT NULL,
  chunk_type text NOT NULL CHECK (
    chunk_type IN (
      'column_definition',
      'value_definition',
      'rule'
    )
  ),
  column_key text,
  table_name text,
  column_name text,
  rule_id text,
  raw_value text,
  text_exact text NOT NULL,
  text_semantic text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding vector(24) NOT NULL,
  UNIQUE (catalog_namespace, chunk_key),
  FOREIGN KEY (catalog_namespace, column_key)
    REFERENCES column_catalog(catalog_namespace, column_key)
    ON DELETE CASCADE,
  FOREIGN KEY (catalog_namespace, rule_id)
    REFERENCES rule_catalog(catalog_namespace, rule_id)
    ON DELETE CASCADE,
  search_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', coalesce(text_exact, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(text_semantic, '')), 'B')
  ) STORED
);

CREATE INDEX IF NOT EXISTS metadata_chunks_search_tsv_idx
  ON metadata_chunks USING GIN (search_tsv);

CREATE INDEX IF NOT EXISTS metadata_chunks_embedding_idx
  ON metadata_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS metadata_chunks_catalog_namespace_idx
  ON metadata_chunks (catalog_namespace);

CREATE INDEX IF NOT EXISTS metadata_chunks_catalog_namespace_chunk_type_idx
  ON metadata_chunks (catalog_namespace, chunk_type);

CREATE INDEX IF NOT EXISTS metadata_chunks_catalog_namespace_column_key_idx
  ON metadata_chunks (catalog_namespace, column_key);

CREATE INDEX IF NOT EXISTS metadata_chunks_catalog_namespace_table_name_idx
  ON metadata_chunks (catalog_namespace, table_name);

CREATE OR REPLACE FUNCTION hybrid_search(
  target_namespace text,
  query_text text,
  query_embedding vector(24),
  top_k integer DEFAULT 10
)
RETURNS TABLE (
  id bigint,
  catalog_namespace text,
  chunk_key text,
  chunk_type text,
  column_key text,
  table_name text,
  column_name text,
  rule_id text,
  raw_value text,
  score double precision,
  lex_rank double precision,
  semantic_distance double precision,
  payload jsonb,
  text_exact text,
  text_semantic text
)
LANGUAGE SQL
STABLE
AS $$
WITH input_query AS (
  SELECT NULLIF(trim(coalesce(query_text, '')), '') AS query_text
),
ft AS (
  SELECT
    CASE
      WHEN input_query.query_text IS NULL THEN NULL
      ELSE
        websearch_to_tsquery('simple', unaccent(input_query.query_text)) ||
        websearch_to_tsquery('english', unaccent(input_query.query_text))
    END AS tsq
  FROM input_query
),
lex AS (
  SELECT
    c.id,
    ts_rank_cd(c.search_tsv, ft.tsq) AS lex_rank,
    row_number() OVER (
      ORDER BY ts_rank_cd(c.search_tsv, ft.tsq) DESC, c.id
    ) AS lex_pos
  FROM metadata_chunks c
  CROSS JOIN ft
  WHERE ft.tsq IS NOT NULL
    AND c.catalog_namespace = target_namespace
    AND c.search_tsv @@ ft.tsq
  ORDER BY lex_rank DESC, c.id
  LIMIT GREATEST(top_k * 5, 20)
),
sem AS (
  SELECT
    c.id,
    c.embedding <=> query_embedding AS semantic_distance,
    row_number() OVER (
      ORDER BY c.embedding <=> query_embedding ASC, c.id
    ) AS sem_pos
  FROM metadata_chunks c
  WHERE c.catalog_namespace = target_namespace
  ORDER BY semantic_distance ASC, c.id
  LIMIT GREATEST(top_k * 5, 20)
),
fused AS (
  SELECT
    COALESCE(lex.id, sem.id) AS id,
    COALESCE(1.0 / (60 + lex.lex_pos), 0.0) +
    COALESCE(1.0 / (60 + sem.sem_pos), 0.0) AS score,
    lex.lex_rank,
    sem.semantic_distance
  FROM lex
  FULL OUTER JOIN sem USING (id)
)
SELECT
  c.id,
  c.catalog_namespace,
  c.chunk_key,
  c.chunk_type,
  c.column_key,
  c.table_name,
  c.column_name,
  c.rule_id,
  c.raw_value,
  fused.score,
  fused.lex_rank,
  fused.semantic_distance,
  c.payload,
  c.text_exact,
  c.text_semantic
FROM fused
JOIN metadata_chunks c ON c.id = fused.id
ORDER BY
  fused.score DESC,
  fused.lex_rank DESC NULLS LAST,
  fused.semantic_distance ASC NULLS LAST,
  c.id
LIMIT top_k;
$$;
