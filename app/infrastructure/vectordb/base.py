"""
Abstract Base Class for Vector Stores.

Defines the interface for all vector store implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Document with text and metadata."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class QueryResult:
    """Result from a vector store query."""

    documents: list[Document]
    scores: list[float] | None = None


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    All vector store implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
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
            collection_type: Type of collection (e.g., 'resume', 'job_posting').
            metadata: Additional metadata to store with the document.

        Returns:
            The document ID.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    async def health_check(self) -> bool:
        """Check if the vector store is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Try a simple query
            await self.query("test", "resumes", n_results=1)
            return True
        except Exception:
            return False

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Get the store name."""
        pass
