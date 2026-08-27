CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- Sparse arm without OpenSearch: Postgres FTS (tsvector + GIN), per §3.4.
