-- Bootstrap extensions and persistent schema objects for the metadata retriever.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Stable source catalog for one owner + namespace scope.
CREATE TABLE IF NOT EXISTS column_catalog (
  resource_owner text NOT NULL,
  resource_namespace text NOT NULL,
  table_name text NOT NULL,
  column_name text NOT NULL,
  raw_column_name text,
  description text NOT NULL,
  aliases text[] NOT NULL DEFAULT '{}',
  data_type text,
  value_cardinality integer NOT NULL DEFAULT 0,
  mandatory_description_in_prompt boolean NOT NULL DEFAULT false,
  PRIMARY KEY (resource_owner, resource_namespace, table_name, column_name)
);

-- Low-cardinality value metadata that belongs to one table-qualified column.
CREATE TABLE IF NOT EXISTS column_value_catalog (
  resource_owner text NOT NULL,
  resource_namespace text NOT NULL,
  table_name text NOT NULL,
  column_name text NOT NULL,
  raw_value text NOT NULL,
  value_description text NOT NULL,
  synonyms text[] NOT NULL DEFAULT '{}',
  business_tags text[] NOT NULL DEFAULT '{}',
  PRIMARY KEY (
    resource_owner,
    resource_namespace,
    table_name,
    column_name,
    raw_value
  ),
  FOREIGN KEY (resource_owner, resource_namespace, table_name, column_name)
    REFERENCES column_catalog(
      resource_owner,
      resource_namespace,
      table_name,
      column_name
    )
    ON DELETE CASCADE
);

-- Reserved rule catalog for future rule-RAG / prompt-augmentation work.
CREATE TABLE IF NOT EXISTS rule_catalog (
  resource_owner text NOT NULL,
  resource_namespace text NOT NULL,
  rule_id text NOT NULL,
  text_exact text NOT NULL,
  text_semantic text NOT NULL,
  candidate_columns jsonb NOT NULL DEFAULT '[]'::jsonb,
  intent text,
  priority integer NOT NULL DEFAULT 0,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (resource_owner, resource_namespace, rule_id)
);

-- Unified retrieval table. The query path retrieves chunks here, then rolls
-- those hits back up to table-qualified columns in Python.
CREATE TABLE IF NOT EXISTS metadata_chunks (
  id bigserial PRIMARY KEY,
  resource_owner text NOT NULL,
  resource_namespace text NOT NULL,
  chunk_key text NOT NULL,
  chunk_type text NOT NULL CHECK (
    chunk_type IN (
      'column_definition',
      'value_definition',
      'rule'
    )
  ),
  table_name text,
  column_name text,
  rule_id text,
  raw_value text,
  text_exact text NOT NULL,
  text_semantic text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding vector(24) NOT NULL,
  UNIQUE (resource_owner, resource_namespace, chunk_key),
  FOREIGN KEY (resource_owner, resource_namespace, table_name, column_name)
    REFERENCES column_catalog(
      resource_owner,
      resource_namespace,
      table_name,
      column_name
    )
    ON DELETE CASCADE,
  FOREIGN KEY (resource_owner, resource_namespace, rule_id)
    REFERENCES rule_catalog(resource_owner, resource_namespace, rule_id)
    ON DELETE CASCADE,
  search_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', coalesce(text_exact, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(text_semantic, '')), 'B')
  ) STORED
);

-- Lexical and vector indexes that support hybrid retrieval.
CREATE INDEX IF NOT EXISTS metadata_chunks_search_tsv_idx
  ON metadata_chunks USING GIN (search_tsv);

CREATE INDEX IF NOT EXISTS metadata_chunks_embedding_idx
  ON metadata_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS metadata_chunks_owner_namespace_idx
  ON metadata_chunks (resource_owner, resource_namespace);

CREATE INDEX IF NOT EXISTS metadata_chunks_owner_namespace_chunk_type_idx
  ON metadata_chunks (resource_owner, resource_namespace, chunk_type);

CREATE INDEX IF NOT EXISTS metadata_chunks_owner_namespace_table_column_idx
  ON metadata_chunks (resource_owner, resource_namespace, table_name, column_name);

CREATE INDEX IF NOT EXISTS metadata_chunks_owner_namespace_table_name_idx
  ON metadata_chunks (resource_owner, resource_namespace, table_name);
