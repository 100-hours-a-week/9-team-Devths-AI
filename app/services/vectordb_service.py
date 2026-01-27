"""
VectorDB Service using ChromaDB

Provides vector storage and retrieval for RAG (Retrieval-Augmented Generation).
"""

import logging
import os
from typing import Any

import chromadb
import google.genai as genai
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorDBService:
    """VectorDB Service for storing and retrieving embeddings"""

    def __init__(self, api_key: str | None = None, persist_directory: str = "./chroma_db"):
        """
        Initialize VectorDB Service

        Args:
            api_key: Google API key for embeddings (uses GOOGLE_API_KEY env var if not provided)
            persist_directory: Directory to persist ChromaDB data
        """
        # Configure Gemini API for embeddings
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        # Initialize Gemini Client for embeddings
        self.genai_client = genai.Client(api_key=api_key)

        # Initialize ChromaDB
        # ChromaDB 0.4.24 uses Client with Settings
        self.chroma_client = chromadb.Client(
            Settings(persist_directory=persist_directory, anonymized_telemetry=False)
        )

        # Collection names
        self.RESUME_COLLECTION = "resumes"
        self.POSTING_COLLECTION = "job_postings"
        self.PORTFOLIO_COLLECTION = "portfolios"

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

        logger.info(f"VectorDB Service initialized with ChromaDB at {persist_directory}")

    def _get_collection(self, collection_type: str):
        """Get collection by type"""
        # Handle both singular and plural forms
        if collection_type in ("resume", "resumes"):
            return self.resume_collection
        elif collection_type in ("job_posting", "job_postings"):
            return self.posting_collection
        elif collection_type in ("portfolio", "portfolios"):
            return self.portfolio_collection
        else:
            raise ValueError(f"Invalid collection type: {collection_type}")

    async def create_embedding(self, text: str) -> list[float]:
        """
        Create embedding using gemini-embedding-001

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            result = self.genai_client.models.embed_content(
                model="gemini-embedding-001", contents=text
            )
            # Extract embedding from EmbedContentResponse
            # result.embeddings is a list of Embedding objects
            if hasattr(result, "embeddings") and len(result.embeddings) > 0:
                return result.embeddings[0].values
            else:
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
        """
        Add document to VectorDB

        Args:
            document_id: Unique document ID
            text: Document text
            collection_type: "resume", "job_posting", or "portfolio"
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

            # Add to collection
            collection.add(
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
            collection_type: "resume", "job_posting", or "portfolio"

        Returns:
            List of vector IDs
        """
        try:
            collection = self._get_collection(collection_type)

            ids = []
            embeddings = []
            texts = []
            metadatas = []

            for doc in documents:
                doc_id = doc["id"]
                text = doc["text"]
                metadata = doc.get("metadata", {})

                # Create embedding
                embedding = await self.create_embedding(text)

                # Prepare metadata
                metadata["document_id"] = doc_id
                metadata["collection_type"] = collection_type

                ids.append(doc_id)
                embeddings.append(embedding)
                texts.append(text)
                metadatas.append(metadata)

            # Add batch to collection
            collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

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
    ) -> list[dict[str, Any]]:
        """
        Query VectorDB for similar documents

        Args:
            query_text: Query text
            collection_type: "resume", "job_posting", or "portfolio"
            n_results: Number of results to return
            where: Metadata filter (optional)

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
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                        }
                    )

            logger.info(f"Query returned {len(formatted_results)} results from {collection_type}")
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
            collection_type: "resume", "job_posting", or "portfolio"

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
            collection_type: "resume", "job_posting", or "portfolio"

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
            collection_type: "resume", "job_posting", or "portfolio"

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
            collection_type: "resume", "job_posting", or "portfolio"

        Returns:
            Number of documents
        """
        try:
            collection = self._get_collection(collection_type)
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
