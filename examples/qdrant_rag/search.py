from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from benchmax.rag.search import SearchResult


class QdrantSearch:
    """Lazy, pickle-safe lexical/vector/hybrid Qdrant search."""

    def __init__(
        self,
        collection_name: str,
        *,
        url: str,
        api_key: str | None = None,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._url = url
        self._api_key = api_key
        self._embed_fn = embed_fn
        self._client: Any = None

    @property
    def available_modes(self) -> list[str]:
        return ["hybrid", "lexical", "vector"] if self._embed_fn else ["lexical"]

    async def search(self, query: str, mode: str = "auto", top_k: int = 10) -> list[SearchResult]:
        from qdrant_client import QdrantClient, models

        if mode == "auto":
            mode = self.available_modes[0]
        if mode not in self.available_modes:
            raise ValueError(
                f"search mode {mode!r} is unavailable. Available: {self.available_modes}"
            )
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 10
        if self._client is None:
            self._client = QdrantClient(url=self._url, api_key=self._api_key)

        bm25 = models.Document(text=query, model="qdrant/bm25")
        if mode == "lexical":
            kwargs: dict[str, Any] = {"query": bm25, "using": "bm25"}
        else:
            vector = (await self._embed_fn([query]))[0]
            kwargs = {"query": vector, "using": "dense"}
            if mode == "hybrid":
                kwargs = {
                    "prefetch": [
                        models.Prefetch(query=bm25, using="bm25", limit=2 * top_k),
                        models.Prefetch(query=vector, using="dense", limit=2 * top_k),
                    ],
                    "query": models.RrfQuery(rrf=models.Rrf(k=60)),
                }
        response = await asyncio.to_thread(
            self._client.query_points, self._collection_name, limit=top_k, **kwargs
        )
        results: list[SearchResult] = []
        for point in response.points:
            metadata = dict(point.payload or {})
            content = metadata.pop("content", "")
            results.append(
                {
                    "content": content,
                    "source": str(metadata.get("file", "")),
                    "metadata": metadata,
                    "score": point.score,
                }
            )
        return results

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_client": None}


__all__ = ["QdrantSearch"]
