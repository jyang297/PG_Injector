-- Retrieval-policy SQL lives separately from bootstrap schema so ranking logic
-- can evolve without bloating the init file.
CREATE OR REPLACE FUNCTION hybrid_search(
  target_owner text,
  target_namespace text,
  query_text text,
  query_embedding vector(24),
  top_k integer DEFAULT 10
)
RETURNS TABLE (
  id bigint,
  resource_owner text,
  resource_namespace text,
  chunk_key text,
  chunk_type text,
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
    AND c.resource_owner = target_owner
    AND c.resource_namespace = target_namespace
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
  WHERE c.resource_owner = target_owner
    AND c.resource_namespace = target_namespace
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
  c.resource_owner,
  c.resource_namespace,
  c.chunk_key,
  c.chunk_type,
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
