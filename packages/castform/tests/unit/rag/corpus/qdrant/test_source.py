from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.qdrant.source import QdrantChunkSource, _to_filter
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    NotPredicate,
    OrPredicate,
)
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse


class FakeQdrant:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls: list[tuple[str, dict]] = []

    def get_collection(self, name):
        raise UnexpectedResponse(404, "Not Found", b"", httpx.Headers())

    def create_collection(self, name, **kwargs):
        self.calls.append(("create_collection", kwargs))

    def upsert(self, name, points):
        self.calls.append(("upsert", {"points": points}))

    def query_points(self, name, **kwargs):
        self.calls.append(("query_points", kwargs))
        return SimpleNamespace(points=self.results.pop(0) if self.results else [])


def _source(results=None, embed_fn=None):
    source = QdrantChunkSource("docs", url="http://qdrant.test", embed_fn=embed_fn)
    source._qdrant = fake = FakeQdrant(results)
    return source, fake


def _point(point_id, content, file="a.md", index=0, score=1.0):
    payload = {"content": content, "_chars": len(content), "file": file, "index": index}
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def _embed(texts):
    return [[float(len(t)), 1.0] for t in texts]


def test_populate_creates_collection_and_points():
    source, fake = _source(embed_fn=_embed)
    chunk = Chunk(content="hello", metadata=(("file", "a.md"), ("index", 3)))
    source.populate_from_chunks(ChunkCollection([chunk]), show_summary=False)

    (_, create), (_, upsert) = fake.calls
    assert create["vectors_config"]["dense"].size == 2
    assert create["sparse_vectors_config"]["bm25"].modifier == models.Modifier.IDF
    (point,) = upsert["points"]
    assert point.payload == {"file": "a.md", "index": 3, "content": "hello", "_chars": 5}
    assert point.vector["bm25"] == models.Document(text="hello", model="qdrant/bm25")
    assert point.vector["dense"] == [5.0, 1.0]


def test_search_modes():
    source, fake = _source(embed_fn=_embed)
    source.search({"mode": "lexical", "top_k": 3, "text_query": "q"})
    source.search({"mode": "vector", "top_k": 3, "vector_query": [1.0, 0.0]})
    source.search(
        {
            "mode": "hybrid",
            "top_k": 3,
            "text_query": "q",
            "vector_query": [1.0, 0.0],
            "hybrid": {"vector_weight": 2.0},
        }
    )
    lexical, vector, hybrid = (kwargs for _, kwargs in fake.calls)
    assert (lexical["using"], lexical["query"].text) == ("bm25", "q")
    assert (vector["using"], vector["query"]) == ("dense", [1.0, 0.0])
    assert [p.using for p in hybrid["prefetch"]] == ["bm25", "dense"]
    assert hybrid["query"].rrf == models.Rrf(k=60, weights=[1.0, 2.0])


def test_vector_search_requires_embed_fn():
    source, _ = _source()
    with pytest.raises(UnsupportedSearchModeError):
        source.search({"mode": "vector", "top_k": 3, "vector_query": [1.0]})


def test_filter_translation():
    flt = _to_filter(
        AndPredicate(
            (
                OrPredicate((FieldPredicate("file", "eq", "a.md"), FieldPredicate("n", "eq", 1))),
                NotPredicate(FieldPredicate("tag", "in", ["x"])),
            )
        )
    )
    either, negated = flt.must
    assert either.should[0].must[0].match == models.MatchValue(value="a.md")
    assert either.should[1].must[0].range == models.Range(gte=1, lte=1)
    assert negated.must_not[0].must[0].match == models.MatchAny(any=["x"])
    with pytest.raises(InvalidFilterError):
        _to_filter(FieldPredicate("tag", "in", "x"))


def test_search_related_dedupes_and_skips_source_and_neighbors():
    source_chunk = Chunk(content="seed", metadata=(("file", "a.md"), ("index", 5)))
    source, _ = _source(
        results=[
            [
                _point("s", "seed", index=5),
                _point("n", "neighbor", index=6),
                _point("far", "far", index=9, score=0.4),
                _point("x", "other file", file="b.md", score=0.2),
            ],
            [_point("far", "far", index=9, score=0.9)],
        ]
    )
    related = source.search_related(source_chunk, ["q1", "q2"])

    assert [r["chunk"].content for r in related] == ["far", "other file"]
    assert related[0]["queries"] == ["q1", "q2"]
    assert related[0]["max_score"] == 0.9
