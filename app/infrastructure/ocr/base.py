"""
Abstract Base Class for OCR Providers.

Defines the interface for all OCR provider implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FileType(str, Enum):
    """Supported file types for OCR."""

    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"


@dataclass
class PageText:
    """Extracted text from a single page."""

    page_number: int
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    pages: list[PageText]
    total_pages: int
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages)


class BaseOCRProvider(ABC):
    """Abstract base class for OCR providers.

    All OCR provider implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    async def extract_text(
        self,
        file_content: bytes,
        file_type: FileType,
        **kwargs: Any,
    ) -> OCRResult:
        """Extract text from a file.

        Args:
            file_content: Raw file content as bytes.
            file_type: Type of file (pdf, png, jpg, etc.).
            **kwargs: Additional provider-specific parameters.

        Returns:
            OCRResult with extracted text from all pages.
        """
        pass

    @abstractmethod
    async def extract_text_from_url(
        self,
        url: str,
        file_type: FileType,
        **kwargs: Any,
    ) -> OCRResult:
        """Extract text from a file URL.

        Args:
            url: URL to the file (e.g., S3 presigned URL).
            file_type: Type of file.
            **kwargs: Additional provider-specific parameters.

        Returns:
            OCRResult with extracted text from all pages.
        """
        pass

    async def health_check(self) -> bool:
        """Check if the provider is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        # Default implementation - subclasses should override if needed
        return True

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass

    @property
    def supported_file_types(self) -> list[FileType]:
        """Get list of supported file types.

        Returns:
            List of supported FileType values.
        """
        return [FileType.PDF, FileType.PNG, FileType.JPG, FileType.JPEG]
