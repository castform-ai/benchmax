# qdrant_rag

an end-to-end Benchmax retrieval-training example backed by Qdrant. the data command chunks Markdown, writes the corpus to a Qdrant collection, and generates grounded train/eval questions. the environment supports lexical, vector, and hybrid retrieval.

## example corpus

place the Markdown documents you want to search over in `documents/`.

if you want to use a dummy corpus, clone the public GitLab Handbook:

```bash
git clone --depth 1 https://gitlab.com/gitlab-com/content-sites/handbook.git documents/gitlab-handbook
```

## build the corpus and dataset

configure Qdrant (a Qdrant Cloud cluster, or any server reachable from the training workers):

```bash
export QDRANT_URL="https://your-cluster.cloud.qdrant.io:6333"
export QDRANT_API_KEY="..."

uv run python main.py data --question-count 20
```

for local experiments, `docker run -p 6333:6333 qdrant/qdrant` and `QDRANT_URL=http://localhost:6333` work for the data command, but hosted validation and training need a URL the platform can reach.

data generation uses Castform's configured LLM endpoint. set `QDRANT_RAG_MODEL_BASE_URL` and `QDRANT_RAG_MODEL_API_KEY` to use another OpenAI-compatible endpoint. `QDRANT_RAG_EMBEDDING_MODEL` and `QDRANT_RAG_QA_MODEL` override the default models.

## validate and launch

```bash
uv run python main.py validate
uv run python main.py launch
```

BM25 runs inside Qdrant (`qdrant/bm25`) and hybrid search fuses BM25 and dense results with RRF inside Qdrant, so the runtime dependencies contain only `qdrant-client`. Castform, chunkers, and ingestion code are excluded.

## tests

```bash
uv run pytest examples/qdrant_rag/tests
```
