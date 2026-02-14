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
        description="Google API key for Gemini (쉼표 구분으로 여러 키 입력 가능)",
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model name",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model name (VectorDB 문서 설계 최종 선정)",
    )

    @property
    def all_google_api_keys(self) -> list[str]:
        """Google API 키를 리스트로 반환 (쉼표 구분 분산 처리)."""
        if not self.google_api_key:
            return []
        return [k.strip() for k in self.google_api_key.split(",") if k.strip()]

    # ============================================
    # vLLM Configuration (GPU Servers)
    # ============================================
    gcp_vllm_base_url: str | None = Field(
        default=None,
        description="vLLM 8B server base URL (평시 질의응답, GCP L4 서버리스)",
    )
    vllm_model_name: str = Field(
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        description="vLLM 8B model name",
    )
    vllm_32b_base_url: str | None = Field(
        default=None,
        description="vLLM 32B server base URL (면접 질문 생성, RunPod H100/A100 서버리스)",
    )
    vllm_32b_model_name: str = Field(
        default="LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
        description="vLLM 32B model name",
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
    easyocr_server_url: str | None = Field(
        default=None,
        description="EasyOCR Server URL (Internal GPU Server, v2)",
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
        default="interview_feedback",
        description="ChromaDB collection for interview Q&A + feedback (문서 설계 A안)",
    )
    chroma_collection_analysis_results: str = Field(
        default="analysis_results",
        description="ChromaDB collection for analysis/matching results",
    )
    chroma_collection_chat_context: str = Field(
        default="chat_context",
        description="ChromaDB collection for important chat context",
    )
    chroma_server_host: str | None = Field(
        default=None,
        description="ChromaDB Server Host (v2 server mode)",
    )
    chroma_server_port: int = Field(
        default=8000,
        description="ChromaDB Server Port",
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
    # OpenAI Configuration (평가 토론용)
    # ============================================
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for evaluation debate",
    )

    # ============================================
    # Evaluation Configuration (면접 답변 분석)
    # ============================================
    eval_gemini_model: str = Field(
        default="gemini-3-pro-preview",
        description="Gemini model for interview evaluation",
    )
    eval_thinking_level: str = Field(
        default="HIGH",
        description="Gemini thinking level for evaluation (NONE, LOW, MEDIUM, HIGH)",
    )
    eval_gpt_model: str = Field(
        default="gpt-4o",
        description="OpenAI model for debate evaluation",
    )
    eval_debate_enabled: bool = Field(
        default=True,
        description="Enable debate feature (requires OpenAI API key)",
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
    # LLM Generation Parameters
    # ============================================
    llm_temperature_chat: float = Field(
        default=0.1,
        description="채팅 응답 temperature (0에 가까울수록 정확, 문서 기반)",
    )
    llm_temperature_analysis: float = Field(
        default=0.1,
        description="분석 temperature (일관성 중요)",
    )
    llm_temperature_interview_question: float = Field(
        default=0.7,
        description="면접 질문 생성 temperature (다양성 필요)",
    )
    llm_temperature_evaluation: float = Field(
        default=0.1,
        description="면접 평가 temperature (정확성 중요)",
    )
    llm_temperature_ocr: float = Field(
        default=0.1,
        description="OCR 텍스트 추출 temperature",
    )
    llm_max_tokens_chat: int = Field(
        default=2048,
        description="채팅 응답 최대 토큰 수",
    )
    llm_max_tokens_analysis: int = Field(
        default=2048,
        description="분석 최대 토큰 수",
    )
    llm_max_tokens_interview: int = Field(
        default=1024,
        description="면접 질문/평가 최대 토큰 수",
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
        """Check if vLLM 8B server is configured."""
        return bool(self.gcp_vllm_base_url)

    @property
    def vllm_32b_available(self) -> bool:
        """Check if vLLM 32B server is configured."""
        return bool(self.vllm_32b_base_url)

    @property
    def langfuse_available(self) -> bool:
        """Check if Langfuse is configured."""
        return bool(self.langfuse_secret_key and self.langfuse_public_key)

    @property
    def openai_available(self) -> bool:
        """Check if OpenAI is configured for debate."""
        return bool(self.openai_api_key)

    @property
    def debate_available(self) -> bool:
        """Check if debate feature is available."""
        return self.eval_debate_enabled and self.openai_available


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
