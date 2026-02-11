"""
ChromaDB Vector Store Implementation.

Implements the BaseVectorStore interface for ChromaDB.
"""

import logging
import os
from typing import Any

import chromadb
import google.genai as genai
from chromadb.config import Settings

from .base import BaseVectorStore, Document, QueryResult

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    # Collection type mappings (문서 [AI] 09_VectorDB_설계.md A안)
    COLLECTION_NAMES = {
        "resume": "resumes",
        "resumes": "resumes",
        "job_posting": "job_postings",
        "job_postings": "job_postings",
        "portfolio": "portfolios",
        "portfolios": "portfolios",
        "interview": "interview_feedback",
        "interview_questions": "interview_feedback",
        "interview_feedback": "interview_feedback",
        "analysis_results": "analysis_results",
        "chat_context": "chat_context",
    }

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        api_key: str | None = None,
        embedding_model: str = "gemini-embedding-001",
        chroma_server_host: str | None = None,
        chroma_server_port: int = 8000,
    ):
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data (embedded mode).
            api_key: Google API key for embeddings.
            embedding_model: Gemini embedding model name.
            chroma_server_host: ChromaDB server host (v2 server mode). If set, use HttpClient.
            chroma_server_port: ChromaDB server port.
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        # Initialize Gemini Client for embeddings
        self.genai_client = genai.Client(api_key=api_key)
        self._embedding_model = embedding_model

        # Initialize ChromaDB: server mode (v2) or embedded mode
        if chroma_server_host:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_server_host,
                port=chroma_server_port,
            )
            logger.info(
                f"ChromaVectorStore initialized (server mode) at {chroma_server_host}:{chroma_server_port}"
            )
        else:
            self.chroma_client = chromadb.Client(
                Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False,
                )
            )
            logger.info(f"ChromaVectorStore initialized at {persist_directory}")

        # Create or get collections (문서 6개 컬렉션)
        self._collections = {
            "resumes": self.chroma_client.get_or_create_collection(
                name="resumes",
                metadata={"description": "Resume embeddings"},
            ),
            "job_postings": self.chroma_client.get_or_create_collection(
                name="job_postings",
                metadata={"description": "Job posting embeddings"},
            ),
            "portfolios": self.chroma_client.get_or_create_collection(
                name="portfolios",
                metadata={"description": "Portfolio embeddings"},
            ),
            "interview_feedback": self.chroma_client.get_or_create_collection(
                name="interview_feedback",
                metadata={
                    "description": "Interview Q&A + feedback (A안: interview_type 메타데이터)"
                },
            ),
            "analysis_results": self.chroma_client.get_or_create_collection(
                name="analysis_results",
                metadata={"description": "Analysis/matching results"},
            ),
            "chat_context": self.chroma_client.get_or_create_collection(
                name="chat_context",
                metadata={"description": "Important chat context"},
            ),
        }

    @property
    def store_name(self) -> str:
        """Get the store name."""
        return "chromadb"

    def _get_collection(self, collection_type: str):
        """Get collection by type.

        Args:
            collection_type: Collection type (e.g., 'resume', 'job_posting').

        Returns:
            ChromaDB collection.

        Raises:
            ValueError: If collection type is invalid.
        """
        collection_name = self.COLLECTION_NAMES.get(collection_type)
        if not collection_name or collection_name not in self._collections:
            raise ValueError(f"Invalid collection type: {collection_type}")
        return self._collections[collection_name]

    async def _create_embedding(self, text: str) -> list[float]:
        """Create embedding using Gemini.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        try:
            result = self.genai_client.models.embed_content(
                model=self._embedding_model,
                contents=text,
            )
            if hasattr(result, "embeddings") and len(result.embeddings) > 0:
                return result.embeddings[0].values
            raise ValueError(f"Unexpected embedding result format: {type(result)}")
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    async def add_document(
        self,
        document_id: str,
        text: str,
        collection_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a document to the vector store.

        Args:
            document_id: Unique identifier for the document.
            text: Document text to embed and store.
            collection_type: Type of collection.
            metadata: Additional metadata.

        Returns:
            The document ID.
        """
        try:
            collection = self._get_collection(collection_type)

            # Create embedding
            embedding = await self._create_embedding(text)

            # Prepare metadata (filter None values - ChromaDB doesn't allow them)
            doc_metadata = metadata or {}
            doc_metadata["document_id"] = document_id
            doc_metadata["collection_type"] = collection_type
            doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}

            # Add to collection
            collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
            )

            logger.info(f"Added document {document_id} to {collection_type}")
            return document_id

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    async def add_documents(
        self,
        documents: list[Document],
        collection_type: str,
    ) -> list[str]:
        """Add multiple documents to the vector store.

        Args:
            documents: List of documents to add.
            collection_type: Type of collection.

        Returns:
            List of document IDs.
        """
        try:
            collection = self._get_collection(collection_type)

            ids = []
            embeddings = []
            texts = []
            metadatas = []

            for doc in documents:
                embedding = await self._create_embedding(doc.text)

                ids.append(doc.id)
                embeddings.append(embedding)
                texts.append(doc.text)

                # Prepare metadata
                doc_metadata = doc.metadata.copy()
                doc_metadata["document_id"] = doc.id
                doc_metadata["collection_type"] = collection_type
                doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
                metadatas.append(doc_metadata)

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(ids)} documents to {collection_type}")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def query(
        self,
        query_text: str,
        collection_type: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Query the vector store for similar documents.

        Args:
            query_text: Query text to search for.
            collection_type: Type of collection to search.
            n_results: Number of results to return.
            where: Filter conditions for metadata.

        Returns:
            QueryResult with matching documents and scores.
        """
        try:
            collection = self._get_collection(collection_type)

            # Create query embedding
            query_embedding = await self._create_embedding(query_text)

            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            # Convert to QueryResult
            documents = []
            scores = []

            if results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = Document(
                        id=doc_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    )
                    documents.append(doc)

                    # ChromaDB returns distances, convert to similarity scores
                    if results["distances"] and len(results["distances"]) > 0:
                        distance = results["distances"][0][i]
                        # Convert distance to similarity (1 / (1 + distance))
                        similarity = 1 / (1 + distance)
                        scores.append(similarity)

            return QueryResult(documents=documents, scores=scores if scores else None)

        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise

    async def get_document(
        self,
        document_id: str,
        collection_type: str,
    ) -> Document | None:
        """Get a specific document by ID.

        Args:
            document_id: Document ID to retrieve.
            collection_type: Type of collection.

        Returns:
            Document if found, None otherwise.
        """
        try:
            collection = self._get_collection(collection_type)

            results = collection.get(ids=[document_id])

            if results["ids"] and len(results["ids"]) > 0:
                return Document(
                    id=results["ids"][0],
                    text=results["documents"][0] if results["documents"] else "",
                    metadata=results["metadatas"][0] if results["metadatas"] else {},
                )

            return None

        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None

    async def get_all_documents_by_user(
        self,
        user_id: str,
        collection_type: str,
    ) -> list[Document]:
        """Get all documents for a specific user.

        Args:
            user_id: User ID to filter by.
            collection_type: Type of collection.

        Returns:
            List of documents for the user.
        """
        try:
            collection = self._get_collection(collection_type)

            results = collection.get(
                where={"user_id": user_id},
            )

            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    doc = Document(
                        id=doc_id,
                        text=results["documents"][i] if results["documents"] else "",
                        metadata=results["metadatas"][i] if results["metadatas"] else {},
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error getting documents by user: {e}")
            return []

    async def delete_document(
        self,
        document_id: str,
        collection_type: str,
    ) -> bool:
        """Delete a document from the vector store.

        Args:
            document_id: Document ID to delete.
            collection_type: Type of collection.

        Returns:
            True if deleted, False if not found.
        """
        try:
            collection = self._get_collection(collection_type)

            # Check if document exists
            existing = collection.get(ids=[document_id])
            if not existing["ids"]:
                return False

            collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id} from {collection_type}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    async def delete_collection(
        self,
        collection_type: str,
    ) -> bool:
        """Delete an entire collection.

        Args:
            collection_type: Type of collection to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            collection_name = self.COLLECTION_NAMES.get(collection_type)
            if not collection_name:
                return False

            self.chroma_client.delete_collection(name=collection_name)

            # Recreate empty collection
            self._collections[collection_name] = self.chroma_client.get_or_create_collection(
                name=collection_name,
            )

            logger.info(f"Deleted collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
