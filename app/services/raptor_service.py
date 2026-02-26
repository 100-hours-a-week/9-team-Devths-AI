"""
ADR-106 Phase 4: RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

문서를 재귀적으로 클러스터링 + 요약하여 트리 구조로 저장.
검색 시 다단계 결과 병합(Level 0 원본 + Level 1~N 요약)으로
추상적 질문과 구체적 질문 모두 대응.

사용 흐름:
1. 문서 업로드 시: build_tree() → ChromaDB에 계층별 저장
2. 검색 시: retrieve_multi_level() → 모든 레벨에서 검색 → (collection_type, Document) 반환
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from app.config.settings import get_settings

if TYPE_CHECKING:
    from app.infrastructure.llm.langchain_wrapper import LangChainLLMGateway
    from app.services.vectordb_service import VectorDBService

logger = logging.getLogger(__name__)


@dataclass
class RaptorNode:
    """RAPTOR 트리의 노드."""

    id: str
    text: str
    level: int  # 0=원본, 1=클러스터요약, 2=상위요약
    children_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RaptorService:
    """ADR-106 Phase 4: RAPTOR 서비스.

    문서 업로드 시 트리를 구성하고,
    검색 시 모든 레벨에서 결과를 반환한다.
    """

    def __init__(
        self,
        vectordb_service: VectorDBService,
        langchain_gateway: LangChainLLMGateway | None = None,
    ):
        self._vectordb = vectordb_service
        self._gateway = langchain_gateway
        settings = get_settings()
        self._max_levels = settings.raptor_max_levels
        self._min_cluster_size = settings.raptor_min_cluster_size
        self._summary_max_docs = settings.raptor_summary_max_docs

    # ========== 트리 구성 ==========

    async def build_tree(
        self,
        texts: list[str],
        collection_name: str,
        user_id: str,
        base_metadata: dict | None = None,
    ) -> dict[str, Any]:
        """문서 텍스트들 → RAPTOR 트리 구성 + ChromaDB 저장.

        Args:
            texts: Level 0 원본 청크 텍스트들 (이미 분할된 상태).
            collection_name: ChromaDB 컬렉션 이름 (예: "resumes").
            user_id: 사용자 ID (필터링용).
            base_metadata: 공통 메타데이터.

        Returns:
            트리 구성 결과 통계.
        """
        if not texts or len(texts) < self._min_cluster_size:
            logger.info(
                "RAPTOR: 청크 수 부족 (%d < %d), 트리 구성 건너뜀",
                len(texts),
                self._min_cluster_size,
            )
            return {"total_nodes": 0, "skipped": True}

        meta = base_metadata or {}
        all_nodes: list[RaptorNode] = []
        current_level_texts = list(texts)

        for level in range(1, self._max_levels + 1):
            if len(current_level_texts) <= self._min_cluster_size:
                logger.info(
                    "RAPTOR: Level %d — 텍스트 %d개, 최소 클러스터 미달. 종료.",
                    level,
                    len(current_level_texts),
                )
                break

            # 1. 클러스터링
            clusters = await self._cluster_texts(current_level_texts)
            if not clusters:
                break

            # 2. 각 클러스터 요약
            summaries: list[str] = []
            for i, cluster_texts in enumerate(clusters):
                summary = await self._summarize_cluster(cluster_texts)
                if summary:
                    node = RaptorNode(
                        id=f"{collection_name}_{user_id}_L{level}_C{i}",
                        text=summary,
                        level=level,
                        metadata={
                            **meta,
                            "raptor_level": level,
                            "user_id": user_id,
                            "cluster_size": len(cluster_texts),
                        },
                    )
                    all_nodes.append(node)
                    summaries.append(summary)

            logger.info(
                "RAPTOR: Level %d — %d개 클러스터 → %d개 요약 생성",
                level,
                len(clusters),
                len(summaries),
            )
            current_level_texts = summaries

        # ChromaDB 저장
        if all_nodes:
            await self._store_nodes(all_nodes, collection_name)

        level_counts = {}
        for node in all_nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1

        result = {
            "total_nodes": len(all_nodes),
            "level_counts": level_counts,
            "skipped": False,
        }
        logger.info("RAPTOR: 트리 구성 완료 — %s", result)
        return result

    async def _cluster_texts(self, texts: list[str]) -> list[list[str]]:
        """텍스트들을 임베딩 유사도 기반 KMeans 클러스터링."""
        try:
            import numpy as np
            from sklearn.cluster import KMeans

            # 임베딩 생성
            embeddings = await self._get_embeddings(texts)
            if not embeddings:
                return []

            matrix = np.array(embeddings)

            # K 결정: sqrt(N), min_cluster_size 이상
            n_clusters = max(2, min(len(texts) // self._min_cluster_size, int(len(texts) ** 0.5)))
            n_clusters = min(n_clusters, len(texts))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(matrix)

            # 같은 클러스터 텍스트 그룹화
            clusters: dict[int, list[str]] = {}
            for label, text in zip(labels, texts, strict=False):
                clusters.setdefault(int(label), []).append(text)

            return list(clusters.values())
        except Exception as e:
            logger.warning("RAPTOR: 클러스터링 실패: %s", e)
            return []

    async def _summarize_cluster(self, texts: list[str]) -> str:
        """클러스터 텍스트들을 LLM으로 요약."""
        if not self._gateway:
            logger.warning("RAPTOR: LangChain gateway 없음, 요약 건너뜀")
            return ""

        # 토큰 제한 위해 최대 N개만
        target_texts = texts[: self._summary_max_docs]
        combined = "\n\n---\n\n".join(target_texts)

        # 전체 길이 제한 (약 4000자 = ~1000토큰)
        if len(combined) > 4000:
            combined = combined[:4000] + "..."

        prompt_text = (
            f"다음 {len(target_texts)}개 텍스트의 핵심 내용을 3-5문장으로 요약하세요.\n"
            "세부 사항보다 전체적인 주제와 핵심 역량/요구사항에 집중하세요.\n\n"
            f"{combined}\n\n요약:"
        )

        try:
            response = await self._gateway.generate(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=500,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            logger.warning("RAPTOR: 요약 실패: %s", e)
            return ""

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Gemini 임베딩 생성."""
        if not self._gateway:
            return []
        try:
            return await self._gateway.create_embeddings(texts)
        except Exception as e:
            logger.warning("RAPTOR: 임베딩 생성 실패: %s", e)
            return []

    async def _store_nodes(self, nodes: list[RaptorNode], collection_name: str) -> None:
        """RAPTOR 노드들을 ChromaDB에 저장."""
        documents = [
            {"id": node.id, "text": node.text, "metadata": node.metadata} for node in nodes
        ]
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._vectordb.add_documents_batch,
                collection_name,
                documents,
            )
            logger.info(
                "RAPTOR: %d개 노드 ChromaDB 저장 완료 (collection=%s)", len(nodes), collection_name
            )
        except Exception as e:
            logger.error("RAPTOR: ChromaDB 저장 실패: %s", e)

    # ========== 다단계 검색 ==========

    async def retrieve_multi_level(
        self,
        query: str,
        collection_name: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[tuple[str, Document]]:
        """모든 RAPTOR 레벨에서 검색하여 (collection_type, Document) 쌍 반환.

        Args:
            query: 검색 쿼리.
            collection_name: ChromaDB 컬렉션 이름.
            user_id: 사용자 ID.
            top_k: 레벨당 최대 검색 결과 수.

        Returns:
            (collection_type, Document) 리스트 — rag_service RRF 병합에 직접 사용 가능.
        """
        all_pairs: list[tuple[str, Document]] = []
        # collection_name에서 collection_type 추출 (resumes → resume)
        ct = collection_name.rstrip("s") if collection_name.endswith("s") else collection_name

        for level in range(1, self._max_levels + 1):
            try:
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda lv=level: self._vectordb.query(
                        collection_name=collection_name,
                        query_text=query,
                        n_results=top_k,
                        where={
                            "$and": [
                                {"raptor_level": {"$eq": lv}},
                                {"user_id": {"$eq": user_id}},
                            ]
                        },
                    ),
                )
                for doc_dict in results:
                    doc = Document(
                        page_content=doc_dict.get("text", doc_dict.get("document", "")),
                        metadata=doc_dict.get("metadata", {}),
                    )
                    all_pairs.append((ct, doc))
            except Exception as e:
                logger.debug(
                    "RAPTOR: Level %d 검색 실패 (collection=%s): %s", level, collection_name, e
                )

        if all_pairs:
            logger.info(
                "RAPTOR: 다단계 검색 — %d개 결과 (collection=%s, levels=1~%d)",
                len(all_pairs),
                collection_name,
                self._max_levels,
            )
        return all_pairs
