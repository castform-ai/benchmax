from __future__ import annotations

import os
from pathlib import Path

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.qdrant.source import QdrantChunkSource
from castform.rag.example_data import (
    DEFAULT_QUESTION_COUNT,
    RagDataModelConfig,
    RagExampleData,
    SyncOpenAIEmbedder,
)
from qdrant_rag_env import COLLECTION

ROOT = Path(__file__).parent
_DRIVER = RagExampleData(name="qdrant-rag", root=ROOT, env_prefix="QDRANT_RAG")
DATA_DIR = _DRIVER.data_dir


def require_dataset_files() -> dict[str, Path]:
    return _DRIVER.require_dataset_files()


def build_chunks() -> ChunkCollection:
    return _DRIVER.build_chunks()


def qdrant_url() -> str:
    url = os.environ.get("QDRANT_URL", "").strip()
    if not url:
        raise RuntimeError("configure QDRANT_URL")
    return url


def qdrant_api_key() -> str | None:
    return os.environ.get("QDRANT_API_KEY", "").strip() or None


def ingest_corpus(chunks: ChunkCollection, config: RagDataModelConfig) -> None:
    embedder = SyncOpenAIEmbedder(config, request_id="qdrant-rag-ingest")
    source = QdrantChunkSource(
        COLLECTION,
        url=qdrant_url(),
        api_key=qdrant_api_key(),
        embed_fn=embedder,
    )
    source.populate_from_chunks(chunks)


def prepare_data(
    *,
    force: bool = False,
    question_count: int = DEFAULT_QUESTION_COUNT,
) -> dict[str, Path]:
    return _DRIVER.prepare(
        ingest_corpus,
        force=force,
        target_questions=question_count,
    )
