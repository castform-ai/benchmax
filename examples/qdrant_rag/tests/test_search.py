from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmax.rag.search import SearchClient
from qdrant_client import models

_SEARCH_PATH = Path(__file__).parents[1] / "search.py"
_SPEC = importlib.util.spec_from_file_location("qdrant_rag_example_search", _SEARCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
QdrantSearch = _MODULE.QdrantSearch


@pytest.mark.asyncio
async def test_hybrid_search_uses_server_side_bm25_and_rrf() -> None:
    point = SimpleNamespace(score=0.8, payload={"content": "answer", "file": "guide.md"})
    calls: list[dict] = []
    client = SimpleNamespace(
        query_points=lambda name, **kwargs: calls.append(kwargs) or SimpleNamespace(points=[point])
    )
    search = QdrantSearch(
        "docs", url="http://qdrant.test", embed_fn=AsyncMock(return_value=[[0.1, 0.2]])
    )
    search._client = client

    results = await search.search("question", top_k=3)

    assert isinstance(search, SearchClient)
    assert results == [
        {"content": "answer", "source": "guide.md", "metadata": {"file": "guide.md"}, "score": 0.8}
    ]
    (call,) = calls
    lexical, dense = call["prefetch"]
    assert lexical.query == models.Document(text="question", model="qdrant/bm25")
    assert dense.query == [0.1, 0.2]
    assert call["query"] == models.RrfQuery(rrf=models.Rrf(k=60))


def test_available_modes_is_pure_and_does_not_initialize_client() -> None:
    search = QdrantSearch("docs", url="http://qdrant.test", embed_fn=AsyncMock())

    assert search.available_modes == ["hybrid", "lexical", "vector"]
    assert search._client is None


@pytest.mark.asyncio
async def test_invalid_mode_fails_before_connecting() -> None:
    search = QdrantSearch("docs", url="http://qdrant.test")

    with pytest.raises(ValueError, match="unavailable"):
        await search.search("question", mode="vector")
    assert search._client is None
