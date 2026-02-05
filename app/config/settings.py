"""
Application Settings using Pydantic Settings.

Centralized configuration management with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================
    # Environment
    # ============================================
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # ============================================
    # API Configuration
    # ============================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # ============================================
    # Google Gemini API
    # ============================================
    google_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model name",
    )

    # ============================================
    # vLLM Configuration (GCP GPU Server)
    # ============================================
    gcp_vllm_base_url: str | None = Field(
        default=None,
        description="GCP vLLM server base URL",
    )
    vllm_model_name: str = Field(
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B",
        description="vLLM model name",
    )

    # ============================================
    # OCR Configuration
    # ============================================
    clova_ocr_api_url: str | None = Field(
        default=None,
        description="Naver CLOVA OCR API URL",
    )
    clova_ocr_secret_key: str | None = Field(
        default=None,
        description="Naver CLOVA OCR secret key",
    )

    # ============================================
    # VectorDB Configuration (ChromaDB)
    # ============================================
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="ChromaDB persistence directory",
    )
    chroma_collection_resume: str = Field(
        default="resumes",
        description="ChromaDB collection for resumes",
    )
    chroma_collection_posting: str = Field(
        default="job_postings",
        description="ChromaDB collection for job postings",
    )
    chroma_collection_portfolio: str = Field(
        default="portfolios",
        description="ChromaDB collection for portfolios",
    )
    chroma_collection_interview: str = Field(
        default="interview_questions",
        description="ChromaDB collection for interview questions",
    )

    # ============================================
    # Redis Configuration
    # ============================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_session_ttl: int = Field(
        default=3600,
        description="Session TTL in seconds (1 hour)",
    )
    redis_task_ttl: int = Field(
        default=86400,
        description="Task TTL in seconds (24 hours)",
    )

    # ============================================
    # Task Queue Configuration
    # ============================================
    task_storage_dir: str = Field(
        default="/tmp/ai_tasks",
        description="File-based task storage directory (development)",
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL",
    )

    # ============================================
    # Langfuse Observability
    # ============================================
    langfuse_secret_key: str | None = Field(
        default=None,
        description="Langfuse secret key",
    )
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL",
    )

    # ============================================
    # RAG Configuration
    # ============================================
    rag_max_context_length: int = Field(
        default=4000,
        description="Maximum context length for RAG (characters)",
    )
    rag_retrieval_k: int = Field(
        default=3,
        description="Number of documents to retrieve",
    )

    # ============================================
    # Interview Configuration
    # ============================================
    interview_max_questions: int = Field(
        default=5,
        description="Maximum number of interview questions",
    )
    interview_max_followup_depth: int = Field(
        default=3,
        description="Maximum follow-up question depth",
    )

    # ============================================
    # Computed Properties
    # ============================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def clova_available(self) -> bool:
        """Check if CLOVA OCR is configured."""
        return bool(self.clova_ocr_api_url and self.clova_ocr_secret_key)

    @property
    def vllm_available(self) -> bool:
        """Check if vLLM server is configured."""
        return bool(self.gcp_vllm_base_url)

    @property
    def langfuse_available(self) -> bool:
        """Check if Langfuse is configured."""
        return bool(self.langfuse_secret_key and self.langfuse_public_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
