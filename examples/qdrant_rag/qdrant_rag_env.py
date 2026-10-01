from __future__ import annotations

from benchmax.envs import InjectedAuth
from benchmax.rag.embed import OpenAIEmbedder
from benchmax.rag.env import RagEnv
from search import QdrantSearch

COLLECTION = "benchmax-rag"
MAX_SEARCH_CALLS = 8


class QdrantRagEnv(RagEnv):
    system_prompt = RagEnv.render_system_prompt(
        corpus_description="the documents indexed in the Qdrant RAG example collection",
        max_search_calls=MAX_SEARCH_CALLS,
    )

    def __init__(
        self,
        *,
        judge_base_url: str,
        embedding_base_url: str,
        url: str,
        api_key: str | None = None,
    ) -> None:
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            base_url=embedding_base_url,
            auth=InjectedAuth("embedding"),
        )
        super().__init__(
            search=QdrantSearch(
                COLLECTION,
                url=url,
                api_key=api_key,
                embed_fn=embedder,
            ),
            judge_base_url=judge_base_url,
            judge_model="gpt-5.4-mini",
            judge_auth=InjectedAuth("judge"),
            max_search_calls=MAX_SEARCH_CALLS,
        )


__all__ = ["QdrantRagEnv"]
