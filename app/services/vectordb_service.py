"""
VectorDB Service using ChromaDB

Provides vector storage and retrieval for RAG (Retrieval-Augmented Generation).
ADR-065: CacheBackedEmbeddings + LocalFileStore 도입으로 임베딩 캐시 적용.
"""

import asyncio
import logging
import os
import random
from typing import Any

import chromadb
import google.genai as genai
from chromadb.config import Settings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

# 임베딩 캐시 디렉터리 (ADR-065)
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache")


class VectorDBService:
    """VectorDB Service for storing and retrieving embeddings"""

    def __init__(
        self,
        api_key: str | None = None,
        persist_directory: str = "./chroma_db",
        chroma_server_host: str | None = None,
        chroma_server_port: int = 8000,
    ):
        """
        Initialize VectorDB Service

        Args:
            api_key: Google API key for embeddings (uses GOOGLE_API_KEY env var if not provided)
            persist_directory: Directory to persist ChromaDB data (embedded mode)
            chroma_server_host: ChromaDB server host (v2 server mode). If set, use HttpClient.
            chroma_server_port: ChromaDB server port
        """
        # Configure Gemini API for embeddings
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 필요합니다.")

        # 쉼표 구분 다중 키 → 각 키별 클라이언트 생성 (분산 처리, 폴백용)
        api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self._genai_clients = [genai.Client(api_key=k) for k in api_keys]
        self.genai_client = self._genai_clients[0]  # 하위 호환
        logger.info(f"VectorDB 임베딩: {len(self._genai_clients)}개 API 키 로드")

        # --- ADR-065: CacheBackedEmbeddings + LocalFileStore ---
        primary_key = api_keys[0]
        self._underlying_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=primary_key,
        )
        cache_dir = os.path.expanduser(EMBEDDING_CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)
        self._embedding_store = LocalFileStore(cache_dir)
        self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self._underlying_embeddings,
            self._embedding_store,
            namespace="gemini-embedding-001",
        )
        logger.info(f"임베딩 캐시 초기화: {cache_dir}")

        # Initialize ChromaDB: server mode (v2) or embedded mode
        if chroma_server_host:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_server_host,
                port=chroma_server_port,
            )
            logger.info(
                f"VectorDB Service initialized (server mode) at {chroma_server_host}:{chroma_server_port}"
            )
        else:
            persist_directory = os.path.expanduser(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"VectorDB Service initialized with ChromaDB at {persist_directory}")

        # Collection names (문서 [AI] 09_VectorDB_설계.md 6개 컬렉션)
        self.RESUME_COLLECTION = "resumes"
        self.POSTING_COLLECTION = "job_postings"
        self.PORTFOLIO_COLLECTION = "portfolios"
        self.INTERVIEW_COLLECTION = "interview_feedback"
        self.ANALYSIS_RESULTS_COLLECTION = "analysis_results"
        self.CHAT_CONTEXT_COLLECTION = "chat_context"

        # Create or get collections
        self.resume_collection = self.chroma_client.get_or_create_collection(
            name=self.RESUME_COLLECTION, metadata={"description": "Resume embeddings"}
        )
        self.posting_collection = self.chroma_client.get_or_create_collection(
            name=self.POSTING_COLLECTION, metadata={"description": "Job posting embeddings"}
        )
        self.portfolio_collection = self.chroma_client.get_or_create_collection(
            name=self.PORTFOLIO_COLLECTION, metadata={"description": "Portfolio embeddings"}
        )
        self.interview_collection = self.chroma_client.get_or_create_collection(
            name=self.INTERVIEW_COLLECTION,
            metadata={"description": "Interview Q&A + feedback (A안: interview_type)"},
        )
        self.analysis_results_collection = self.chroma_client.get_or_create_collection(
            name=self.ANALYSIS_RESULTS_COLLECTION,
            metadata={"description": "Analysis/matching results"},
        )
        self.chat_context_collection = self.chroma_client.get_or_create_collection(
            name=self.CHAT_CONTEXT_COLLECTION,
            metadata={"description": "Important chat context"},
        )

    def _get_collection(self, collection_type: str):
        """Get collection by type (문서 6개 컬렉션 + 별칭)"""
        if collection_type in ("resume", "resumes"):
            return self.resume_collection
        elif collection_type in ("job_posting", "job_postings"):
            return self.posting_collection
        elif collection_type in ("portfolio", "portfolios"):
            return self.portfolio_collection
        elif collection_type in ("interview", "interview_questions", "interview_feedback"):
            return self.interview_collection
        elif collection_type == "analysis_results":
            return self.analysis_results_collection
        elif collection_type == "chat_context":
            return self.chat_context_collection
        else:
            raise ValueError(f"Invalid collection type: {collection_type}")

    async def create_embedding(self, text: str) -> list[float]:
        """
        Create embedding using gemini-embedding-001 (ADR-065: 캐시 적용)

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            # CacheBackedEmbeddings: 캐시 히트 시 API 호출 생략 (asyncio.to_thread로 비동기 래핑)
            vectors = await asyncio.to_thread(self.cached_embedder.embed_documents, [text])
            return vectors[0]
        except Exception:
            # 캐시 실패 시 기존 genai.Client 폴백
            logger.warning("캐시 임베딩 실패, genai.Client 폴백 사용")
            return await self._create_embedding_fallback(text)

    async def _create_embedding_fallback(self, text: str) -> list[float]:
        """genai.Client 직접 호출 폴백 (다중 키 라운드로빈)"""
        try:
            client = random.choice(self._genai_clients)
            result = client.models.embed_content(model="gemini-embedding-001", contents=text)
            if hasattr(result, "embeddings") and len(result.embeddings) > 0:
                return result.embeddings[0].values
            else:
                raise ValueError(f"Unexpected embedding result format: {type(result)}")
        except Exception as e:
            logger.error(f"Error creating embedding (fallback): {e}")
            raise

    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for multiple texts (ADR-065: 캐시 적용)

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (same order as texts).
        """
        if not texts:
            return []
        try:
            # CacheBackedEmbeddings: 개별 텍스트별 캐시 히트/미스 처리 (asyncio.to_thread로 비동기 래핑)
            return await asyncio.to_thread(self.cached_embedder.embed_documents, texts)
        except Exception:
            # 캐시 실패 시 기존 genai.Client 폴백
            logger.warning("캐시 배치 임베딩 실패, genai.Client 폴백 사용")
            return await self._create_embeddings_fallback(texts)

    async def _create_embeddings_fallback(self, texts: list[str]) -> list[list[float]]:
        """genai.Client 직접 배치 호출 폴백 (다중 키 라운드로빈)"""
        if not texts:
            return []
        try:
            client = random.choice(self._genai_clients)
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
            if hasattr(result, "embeddings") and result.embeddings:
                return [e.values for e in result.embeddings]
            raise ValueError(f"Unexpected embedding result format: {type(result)}")
        except Exception as e:
            logger.error(f"Error creating batch embeddings (fallback): {e}")
            raise

    async def add_document(
        self,
        document_id: str,
        text: str,
        collection_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add document to VectorDB

        Args:
            document_id: Unique document ID
            text: Document text
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"
            metadata: Additional metadata

        Returns:
            Vector ID
        """
        try:
            collection = self._get_collection(collection_type)

            # Create embedding
            embedding = await self.create_embedding(text)

            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata["document_id"] = document_id
            doc_metadata["collection_type"] = collection_type

            # ChromaDB는 None 값을 허용하지 않음 - 필터링
            doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}

            # Add to collection
            collection.upsert(
                ids=[document_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
            )

            logger.info(f"Added document {document_id} to {collection_type} collection")
            return document_id

        except Exception as e:
            logger.error(f"Error adding document to VectorDB: {e}")
            raise

    async def add_documents_batch(
        self, documents: list[dict[str, Any]], collection_type: str
    ) -> list[str]:
        """
        Add multiple documents to VectorDB (for chunked documents)

        Args:
            documents: List of {
                "id": str,
                "text": str,
                "metadata": dict (optional)
            }
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"

        Returns:
            List of vector IDs
        """
        try:
            collection = self._get_collection(collection_type)

            ids = []
            texts = []
            metadatas = []

            for doc in documents:
                doc_id = doc["id"]
                text = doc["text"]
                metadata = doc.get("metadata", {}).copy()
                metadata["document_id"] = doc_id
                metadata["collection_type"] = collection_type
                metadata = {k: v for k, v in metadata.items() if v is not None}
                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)

            # Batch embed all texts in one API call
            embeddings = await self.create_embeddings(texts)

            # Add batch to collection
            collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

            logger.info(f"Added {len(ids)} documents to {collection_type} collection")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents batch to VectorDB: {e}")
            raise

    async def query(
        self,
        query_text: str,
        collection_type: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query VectorDB for similar documents

        Args:
            query_text: Query text
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"
            n_results: Number of results to return
            where: Metadata filter (optional)
            max_distance: Maximum distance threshold (optional). Results with
                distance > max_distance are filtered out. ChromaDB uses L2
                distance by default (lower = more similar).

        Returns:
            List of results with text and metadata
        """
        try:
            collection = self._get_collection(collection_type)

            # Create query embedding
            query_embedding = await self.create_embedding(query_text)

            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]
                    # max_distance 필터: 임계값 초과 시 저품질 결과 제거
                    if max_distance is not None and distance > max_distance:
                        continue
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": distance,
                        }
                    )

            if max_distance is not None:
                logger.info(
                    "Query returned %d results from %s (max_distance=%.2f)",
                    len(formatted_results),
                    collection_type,
                    max_distance,
                )
            else:
                logger.info(
                    "Query returned %d results from %s",
                    len(formatted_results),
                    collection_type,
                )
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying VectorDB: {e}")
            raise

    async def get_all_documents_by_user(
        self, user_id: str, collection_type: str
    ) -> list[dict[str, Any]]:
        """
        Get all documents for a specific user

        Args:
            user_id: User ID
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"

        Returns:
            List of all documents for the user
        """
        try:
            collection = self._get_collection(collection_type)

            # Get all documents with matching user_id
            result = collection.get(where={"user_id": user_id}, include=["documents", "metadatas"])

            formatted_results = []
            if result and result["ids"] and len(result["ids"]) > 0:
                for i in range(len(result["ids"])):
                    formatted_results.append(
                        {
                            "id": result["ids"][i],
                            "text": result["documents"][i],
                            "metadata": result["metadatas"][i],
                        }
                    )

            from app.utils.log_sanitizer import sanitize_log_input

            logger.info(
                f"Retrieved {len(formatted_results)} documents for user {sanitize_log_input(user_id)} from {collection_type}"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving documents for user: {e}")
            return []

    async def get_document(self, document_id: str, collection_type: str) -> dict[str, Any] | None:
        """
        Get document by ID

        Args:
            document_id: Document ID
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"

        Returns:
            Document data or None if not found
        """
        try:
            collection = self._get_collection(collection_type)

            result = collection.get(
                ids=[document_id], include=["documents", "metadatas", "embeddings"]
            )

            if result and result["ids"] and len(result["ids"]) > 0:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                    "embedding": result["embeddings"][0] if result.get("embeddings") else None,
                }

            return None

        except Exception as e:
            logger.error(f"Error getting document from VectorDB: {e}")
            return None

    async def delete_document(self, document_id: str, collection_type: str) -> bool:
        """
        Delete document from VectorDB

        Args:
            document_id: Document ID
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"

        Returns:
            True if deleted, False otherwise
        """
        try:
            collection = self._get_collection(collection_type)
            collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id} from {collection_type}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document from VectorDB: {e}")
            return False

    def get_collection_count(self, collection_type: str) -> int:
        """
        Get number of documents in collection

        Args:
            collection_type: "resume", "job_posting", "portfolio", "interview_feedback", "analysis_results", "chat_context"

        Returns:
            Number of documents
        """
        try:
            collection = self._get_collection(collection_type)
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        user_id: int | str,
        collection_name: str,
    ) -> list[str]:
        """
        Add multiple texts to VectorDB (for interview dataset embedding)

        Args:
            texts: List of texts to embed
            metadatas: List of metadata dicts
            ids: List of unique IDs
            user_id: User ID (0 for public data)
            collection_name: Collection name (e.g., "interview_feedback")

        Returns:
            List of added IDs
        """
        try:
            collection = self._get_collection(collection_name)

            # Batch embed all texts in one API call
            embeddings = await self.create_embeddings(texts)

            filtered_metadatas = []
            for i in range(len(texts)):
                meta = metadatas[i].copy() if i < len(metadatas) else {}
                meta["user_id"] = str(user_id)
                meta = {k: v for k, v in meta.items() if v is not None}
                filtered_metadatas.append(meta)

            # Add batch to collection
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=filtered_metadatas,
            )

            logger.info(f"Added {len(ids)} texts to {collection_name} collection")
            return ids

        except Exception as e:
            logger.error(f"Error adding texts to VectorDB: {e}")
            raise
