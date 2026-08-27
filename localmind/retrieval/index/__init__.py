"""ANN index engineering — pgvector (HNSW, binary quantization + rescoring, filtered ANN) and
Qdrant as a comparison arm.

`pgvector.py` also ships `SimpleHNSW`, a from-scratch HNSW implementation used to generate the
recall-vs-latency and filtered-ANN measurements *locally*, since Postgres is not running in
this environment (see CONVENTIONS.md: "Postgres/pgvector is NOT running"). `PgVectorIndex` is
the real production path (psycopg + the pgvector extension, lazy-imported, tests marked
`@pytest.mark.docker`) and issues the equivalent SQL.
"""

from __future__ import annotations
