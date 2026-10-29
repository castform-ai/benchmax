from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import cached_property
from typing import Any

from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    InvalidSearchSpecError,
    UnsupportedFilterError,
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    FilterPredicate,
    HybridOptions,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
    SearchMode,
    SearchSpec,
    validate_search_spec_shape,
)

DENSE = "dense"
BM25 = "bm25"
BM25_MODEL = "qdrant/bm25"
CONTENT = "content"
CHARS = "_chars"

EmbedFn = Callable[[list[str]], list[list[float]]]


class QdrantChunkSource:
    """ChunkSource backed by a Qdrant collection.

    Lexical search uses Qdrant's server-side BM25. Vector and hybrid search
    are available when ``embed_fn`` is given.

    Example:
        >>> source = QdrantChunkSource("my-docs", url="http://localhost:6333", embed_fn=embed)
        >>> source.populate_from_folder("./docs")
    """

    def __init__(
        self,
        collection_name: str,
        url: str,
        api_key: str | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.embed_fn = embed_fn
        modes: set[SearchMode] = {"lexical", "vector", "hybrid"} if embed_fn else {"lexical"}
        self._capabilities: SearchCapabilities = {
            "backend": "qdrant",
            "modes": modes,
            "filter_ops": {"field": {"eq", "in", "gte", "lte"}, "logical": {"and", "or", "not"}},
            "ranking": {"bm25", "cosine", "rrf"} if embed_fn else {"bm25"},
            "constraints": {},
            "graph_expansion": False,
        }

    @cached_property
    def _qdrant(self) -> Any:
        from qdrant_client import QdrantClient

        return QdrantClient(url=self.url, api_key=self.api_key, timeout=60)

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_qdrant"}

    def _or_empty(self, call: Callable[[], Any], empty: Any) -> Any:
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            return call()
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return empty
            raise

    def populate_from_folder(
        self,
        docs_path: str,
        embed_fn: EmbedFn | None = None,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Chunk Markdown documents in a folder and upload them."""
        from castform.rag.chunkers.markdown import MarkdownChunker

        if file_extensions is None:
            file_extensions = [".md", ".mdx"]
        chunker = MarkdownChunker(
            min_char=min_chars, max_char=max_chars, chunk_overlap=overlap_chars
        )
        collection = chunker.chunk_folder(docs_path, file_extensions=file_extensions)
        self.populate_from_chunks(collection, embed_fn, batch_size, show_summary)

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        embed_fn: EmbedFn | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a ChunkCollection, creating the collection if needed."""
        from qdrant_client import models

        embed_fn = embed_fn or self.embed_fn
        chunks = list(collection)
        for chunk in chunks:
            if CONTENT in chunk.metadata_dict or CHARS in chunk.metadata_dict:
                raise ValueError(
                    f"chunk metadata may not use the reserved keys {CONTENT!r}, {CHARS!r}"
                )

        has_dense = self._or_empty(
            lambda: (
                DENSE
                in (self._qdrant.get_collection(self.collection_name).config.params.vectors or {})
            ),
            None,
        )
        if chunks and has_dense is not None and has_dense != bool(embed_fn):
            raise ValueError(
                f"collection {self.collection_name!r} was created "
                f"{'with' if has_dense else 'without'} dense vectors; populate it "
                f"{'with' if has_dense else 'without'} an embed_fn"
            )

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            dense = embed_fn([chunk.content for chunk in batch]) if embed_fn else None
            if dense is not None and len(dense) != len(batch):
                raise ValueError(f"embed_fn returned {len(dense)} vectors for {len(batch)} texts")
            if has_dense is None:
                vectors_config = {}
                if dense:
                    vectors_config = {
                        DENSE: models.VectorParams(size=len(dense[0]), distance="Cosine")
                    }
                self._qdrant.create_collection(
                    self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config={BM25: models.SparseVectorParams(modifier="idf")},
                )
                has_dense = dense is not None
            points = []
            for i, chunk in enumerate(batch):
                vector: dict[str, Any] = {
                    BM25: models.Document(text=chunk.content, model=BM25_MODEL)
                }
                if dense:
                    vector[DENSE] = dense[i]
                payload = {**chunk.metadata_dict, CONTENT: chunk.content, CHARS: len(chunk.content)}
                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.hash))
                points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
            self._qdrant.upsert(self.collection_name, points=points)
        if show_summary:
            print(f"Uploaded {len(chunks)} chunks to Qdrant collection {self.collection_name!r}.")

    def get_chunk_count(self) -> int:
        return self._or_empty(lambda: self._qdrant.count(self.collection_name, exact=True).count, 0)

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        from qdrant_client import models

        if n <= 0:
            return []
        length = models.FieldCondition(key=CHARS, range=models.Range(gte=min_chars))
        points = self._or_empty(
            lambda: (
                self._qdrant.query_points(
                    self.collection_name,
                    query=models.SampleQuery(sample=models.Sample.RANDOM),
                    query_filter=models.Filter(must=[length]) if min_chars > 0 else None,
                    limit=n,
                ).points
            ),
            [],
        )
        return [_to_chunk(p) for p in points]

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        context = {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": "",
            "next_chunk_preview": "",
        }
        file, index = chunk.get_metadata("file"), chunk.get_metadata("index")
        if file is None:
            return context
        siblings = sorted(self._scroll(FieldPredicate("file", "eq", file)), key=_position)
        position = next((i for i, c in enumerate(siblings) if c.hash == chunk.hash), None)
        if position is None and _is_int(index):
            position = next(
                (i for i, c in enumerate(siblings) if c.get_metadata("index") == index), None
            )
        prev = siblings[position - 1] if position else None
        nxt = (
            siblings[position + 1]
            if position is not None and position + 1 < len(siblings)
            else None
        )
        context["prev_chunk_preview"] = (
            prev.chunk_str(max_chars=max_chars, truncate="leading")
            if prev
            else "(No previous chunk)"
        )
        context["next_chunk_preview"] = (
            nxt.chunk_str(max_chars=max_chars, truncate="trailing") if nxt else "(No next chunk)"
        )
        return context

    def get_top_level_chunks(self) -> list[Chunk]:
        files = {c.get_metadata("file") for c in self._scroll()} - {None}
        if not files:
            return []
        depth = min(str(f).count("/") for f in files)
        top = [f for f in files if str(f).count("/") == depth]
        chunks = self._scroll(FieldPredicate("file", "in", top))
        return sorted(chunks, key=lambda c: (str(c.get_metadata("file")), _position(c)))

    def _scroll(self, predicate: FilterPredicate | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self._or_empty(
                lambda offset=offset: self._qdrant.scroll(
                    self.collection_name,
                    scroll_filter=_to_filter(predicate),
                    limit=1000,
                    offset=offset,
                ),
                ([], None),
            )
            chunks.extend(_to_chunk(p) for p in points)
            if offset is None:
                return chunks

    def get_search_capabilities(self) -> SearchCapabilities:
        return self._capabilities

    def embed_query(self, text: str) -> list[float] | None:
        return self.embed_fn([text])[0] if self.embed_fn else None

    def search(self, spec: SearchSpec) -> list[Chunk]:
        return [_to_chunk(p) for p in self._query(spec)]

    def search_content(self, spec: SearchSpec) -> list[str]:
        return [_to_chunk(p).content for p in self._query(spec)]

    def search_text(
        self, text_query: str, top_k: int = 10, filter: FilterPredicate | None = None
    ) -> list[Chunk]:
        return self.search(
            {"mode": "lexical", "top_k": top_k, "text_query": text_query, "filter": filter}
        )

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        if not queries:
            return []
        mode = mode or "lexical"
        vectors = self.embed_fn(queries) if mode != "lexical" and self.embed_fn else None
        file, index = source.get_metadata("file"), source.get_metadata("index")
        related: dict[str, dict] = {}
        for i, query in enumerate(queries):
            spec: SearchSpec = {"mode": mode, "top_k": top_k, "text_query": query, "hybrid": hybrid}
            if vectors:
                spec["vector_query"] = vectors[i]
            for point in self._query(spec):
                chunk = _to_chunk(point)
                same_file = file is not None and chunk.get_metadata("file") == file
                other = chunk.get_metadata("index")
                adjacent = (
                    same_file and _is_int(index) and _is_int(other) and abs(other - index) <= 1
                )
                if chunk.content == source.content or adjacent:
                    continue
                entry = related.setdefault(
                    str(point.id),
                    {
                        "chunk": chunk,
                        "queries": [],
                        "same_file": same_file,
                        "max_score": point.score,
                    },
                )
                entry["queries"].append(query)
                entry["max_score"] = max(entry["max_score"], point.score)
        return sorted(
            related.values(),
            key=lambda r: (len(r["queries"]), not r["same_file"], r["max_score"]),
            reverse=True,
        )

    def _query(self, spec: SearchSpec) -> list[Any]:
        from qdrant_client import models

        mode = spec.get("mode")
        if mode not in self._capabilities["modes"]:
            raise UnsupportedSearchModeError("qdrant", str(mode), set(self._capabilities["modes"]))
        errors = validate_search_spec_shape(spec)
        if errors:
            raise InvalidSearchSpecError("qdrant", "; ".join(errors), spec)

        top_k = spec["top_k"]
        flt = _to_filter(spec.get("filter"))
        bm25 = models.Document(text=spec.get("text_query") or "", model=BM25_MODEL)
        vector = spec.get("vector_query")
        if mode == "lexical":
            kwargs: dict[str, Any] = {"query": bm25, "using": BM25}
        elif mode == "vector":
            kwargs = {"query": vector, "using": DENSE}
        else:
            weights = spec.get("hybrid") or {}
            kwargs = {
                "prefetch": [
                    models.Prefetch(query=bm25, using=BM25, limit=2 * top_k, filter=flt),
                    models.Prefetch(query=vector, using=DENSE, limit=2 * top_k, filter=flt),
                ],
                "query": models.RrfQuery(
                    rrf=models.Rrf(
                        k=60,
                        weights=[
                            weights.get("lexical_weight", 1.0),
                            weights.get("vector_weight", 1.0),
                        ],
                    )
                ),
            }
        return self._qdrant.query_points(
            self.collection_name, query_filter=flt, limit=top_k, **kwargs
        ).points


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _position(chunk: Chunk) -> tuple[int, int]:
    index = chunk.get_metadata("index")
    return (0, index) if _is_int(index) else (1, 0)


def _to_chunk(point: Any) -> Chunk:
    payload = dict(point.payload or {})
    content = payload.pop(CONTENT, "")
    payload.pop(CHARS, None)
    return Chunk(content=content, metadata=tuple(payload.items()))


def _to_filter(predicate: FilterPredicate | None) -> Any:
    from qdrant_client import models

    if predicate is None:
        return None
    if isinstance(predicate, (AndPredicate, OrPredicate)):
        if not predicate.clauses:
            raise InvalidFilterError("qdrant", "'and'/'or' needs at least one clause", predicate)
        parts = [_to_filter(c) for c in predicate.clauses]
        if isinstance(predicate, AndPredicate):
            return models.Filter(must=parts)
        return models.Filter(should=parts)
    if isinstance(predicate, NotPredicate):
        return models.Filter(must_not=[_to_filter(predicate.clause)])
    return models.Filter(must=[_field_condition(predicate)])


def _field_condition(predicate: FieldPredicate) -> Any:
    from qdrant_client import models

    key, op, value = predicate.field, predicate.op, predicate.value
    if not isinstance(key, str) or not key.strip():
        raise InvalidFilterError("qdrant", "field name must be a non-empty string", predicate)
    if op == "eq":
        return _equals(predicate, value)
    if op == "in":
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise InvalidFilterError("qdrant", "'in' needs a list of values", predicate)
        values = list(value)
        if all(isinstance(v, str) for v in values):
            return models.FieldCondition(key=key, match=models.MatchAny(any=values))
        return models.Filter(should=[_equals(predicate, v) for v in values])
    if op in ("gte", "lte"):
        if not _is_number(value):
            raise InvalidFilterError("qdrant", f"'{op}' needs a number", predicate)
        return models.FieldCondition(key=key, range=models.Range(**{op: value}))
    raise UnsupportedFilterError("qdrant", f"field operator '{op}' is not supported", predicate)


def _equals(predicate: FieldPredicate, value: Any) -> Any:
    from qdrant_client import models

    if _is_number(value):
        return models.FieldCondition(key=predicate.field, range=models.Range(gte=value, lte=value))
    if isinstance(value, (str, bool)):
        return models.FieldCondition(key=predicate.field, match=models.MatchValue(value=value))
    raise InvalidFilterError("qdrant", f"cannot match a {type(value).__name__} value", predicate)
