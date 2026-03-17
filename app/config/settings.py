"""
Application Settings using Pydantic Settings.

Centralized configuration management with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
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

    @model_validator(mode="after")
    def set_defaults_by_env(self):
        """Enforce defaults based on environment."""
        if self.environment == "development":
            # Dev: Debug True, Log Level DEBUG by default
            if self.debug is False:  # Only override if default
                self.debug = True
            if self.log_level == "INFO":  # Only override if default
                self.log_level = "DEBUG"
        else:
            # Non-Dev: Debug False, Log Level INFO (Safety)
            self.debug = False
            if self.log_level == "DEBUG":
                self.log_level = "INFO"
        return self

    # ============================================
    # API Configuration
    # ============================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # ============================================
    # CORS
    # ============================================
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description=(
            "CORS 허용 origin 목록. "
            "환경변수 ALLOWED_ORIGINS=https://dev.devths.com,http://localhost:3000 처럼 "
            "쉼표 구분 문자열로도 입력 가능."
        ),
    )

    # ============================================
    # Google Gemini API
    # ============================================
    google_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini (쉼표 구분으로 여러 키 입력 가능)",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model name",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model name (VectorDB 문서 설계 최종 선정)",
    )

    # ============================================
    # RAG / MMR Retrieval (RAGChain.retrieve_context)
    # ============================================
    rag_retrieval_k: int = Field(
        default=5,
        description="Number of documents to retrieve per collection (MMR). Phase 2: 3→5 (청크 단위 검색에 맞춤)",
    )
    rag_fetch_k: int = Field(
        default=30,
        description="MMR candidate pool size per collection. Phase 2: 20→30",
    )
    rag_lambda_mult: float = Field(
        default=0.5,
        description="MMR diversity (0=diverse, 1=relevant). 0.5 balanced",
    )
    # ADR-069: 앙상블 리트리버 (희소 BM25 + 밀집 MMR)
    rag_use_bm25: bool = Field(
        default=True,
        description="Use BM25 sparse retriever in ensemble with MMR (ADR-069)",
    )
    rag_ensemble_dense_weight: float = Field(
        default=0.7,
        description="Weight for dense (MMR) retriever in ensemble (0~1). Sparse weight = 1 - this.",
    )
    rag_ensemble_sparse_weight: float = Field(
        default=0.3,
        description="Weight for sparse (BM25) retriever in ensemble (0~1)",
    )
    rag_rerank_top_k: int = Field(
        default=10,
        description="Max number of documents after merge (0 = no limit). ADR-069 long-context rerank.",
    )
    # ADR-104: FlashRank 리랭커 (ONNX 기반 의미적 재정렬)
    rag_use_flashrank: bool = Field(
        default=False,
        description="Use FlashRank reranker for semantic reranking after RRF merge. ADR-104.",
    )
    rag_flashrank_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="FlashRank model name. ADR-104. Options: ms-marco-MiniLM-L-12-v2, ms-marco-MultiBERT-L-12, rank-T5-flan.",
    )
    # ADR-069 후속: BM25 인덱스 상한(메모리·지연 완화), 문서 압축기
    rag_bm25_max_docs: int = Field(
        default=1000,
        description="Max documents to build BM25 index from (0 = no limit). ADR-069 follow-up.",
    )
    rag_use_compressor: bool = Field(
        default=False,
        description="Use LLM document compressor on merged results (ADR-069 Phase 3). Increases latency/cost.",
    )
    rag_compressor_top_k: int = Field(
        default=5,
        description="Number of docs to compress when rag_use_compressor=True. Use smaller k to limit LLM calls.",
    )
    # ADR-077: 다중 쿼리 리트리버 (NORMAL 모드 일반 채팅 시 쿼리 다각화)
    rag_use_multi_query: bool = Field(
        default=False,
        description="Use MultiQueryRetriever for NORMAL chat mode (expands query to multiple perspectives). ADR-077.",
    )
    rag_multi_query_count: int = Field(
        default=3,
        description="Number of perspective queries to generate for MultiQueryRetriever. ADR-077.",
    )
    # ADR-080: 시간 가중 벡터 스토어 리트리버 (interview_feedback 최신성 반영)
    rag_use_time_weighted: bool = Field(
        default=False,
        description="Apply time-decay scoring to interview_feedback retrieval. ADR-080.",
    )
    rag_decay_rate_technical: float = Field(
        default=0.1,
        description="Decay rate for technical questions (higher = faster decay). ADR-080.",
    )
    rag_decay_rate_personality: float = Field(
        default=0.01,
        description="Decay rate for personality questions (lower = slower decay). ADR-080.",
    )
    rag_decay_rate_general: float = Field(
        default=0.05,
        description="Decay rate for general queries. ADR-080.",
    )
    # ADR-078: 다중 벡터 스토어 리트리버 (요약 임베딩 / 가설 쿼리)
    rag_use_multi_vector: bool = Field(
        default=False,
        description="Use MultiVectorRetriever for trend_data (summary or hypothetical). ADR-078.",
    )
    rag_multi_vector_strategy: str = Field(
        default="summary",
        description="MultiVector strategy: 'summary' (요약 임베딩) or 'hypothetical' (가설 쿼리). ADR-078.",
    )
    rag_hypothetical_query_count: int = Field(
        default=3,
        description="Number of hypothetical queries per document for MultiVectorRetriever. ADR-078.",
    )
    # ADR-079: 셀프 쿼리 리트리버 (메타데이터 기반 구조화 검색)
    rag_use_self_query: bool = Field(
        default=False,
        description="Use SelfQueryRetriever for trend_data metadata filtering. ADR-079.",
    )
    # ADR-076: 부모 문서 리트리버 (자식 청크 정밀 검색 → 부모 청크 반환)
    rag_use_parent_retriever: bool = Field(
        default=False,
        description="resumes/portfolios에서 자식 청크(400자)로 정밀 검색 후 부모 청크(2000자) 반환. ADR-076.",
    )
    rag_child_chunk_size: int = Field(
        default=400,
        description="ParentDocumentRetriever 자식 청크 크기(문자 기준). ADR-076.",
    )
    rag_child_chunk_overlap: int = Field(
        default=50,
        description="ParentDocumentRetriever 자식 청크 오버랩(문자 기준). ADR-076.",
    )

    # ============================================
    # ADR-094: Trend Crawling Configuration
    # ============================================
    trend_crawl_urls: str = Field(
        default="",
        description="Comma-separated list of URLs to crawl for trend data. ADR-094.",
    )
    trend_crawl_collection: str = Field(
        default="trend_data",
        description="ChromaDB collection name for trend data. ADR-094.",
    )
    trend_crawl_cron_minute: str = Field(
        default="0",
        description="Celery Beat cron minute for trend crawling. ADR-094.",
    )
    trend_crawl_cron_hour: str = Field(
        default="9",
        description="Celery Beat cron hour for trend crawling. ADR-094.",
    )
    trend_crawl_cron_day_of_week: str = Field(
        default="1",
        description="Celery Beat cron day of week (0=Sun, 1=Mon, ...). ADR-094.",
    )

    # ============================================
    # ADR-101: Tavily Search API (Phase 2 크롤링)
    # ============================================
    tavily_api_key: str = Field(
        default="",
        description="Tavily Search API 키. 미설정 시 Tavily 기능 비활성화. ADR-101.",
    )
    tavily_search_depth: str = Field(
        default="advanced",
        description="Tavily 검색 깊이 (basic/advanced). advanced는 전체 본문 추출. ADR-101.",
    )
    tavily_max_results: int = Field(
        default=5,
        description="Tavily 검색당 최대 결과 수. ADR-101.",
    )
    tavily_days: int = Field(
        default=90,
        ge=0,
        description=(
            "Tavily 검색 결과를 최근 N일 이내로 제한. 0이면 제한 없음. ADR-131. "
            "예: 90 → 90일 이내 발행된 문서만 반환."
        ),
    )
    tavily_min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Tavily 결과 최소 relevance score (0.0~1.0). 미달 결과 제거. ADR-131. "
            "0.0으로 설정 시 score 필터 비활성화."
        ),
    )
    tavily_min_content_length: int = Field(
        default=200,
        ge=0,
        description=(
            "Tavily 결과 최소 본문 길이(문자 수). 미달 결과 제거. ADR-131. "
            "0으로 설정 시 길이 필터 비활성화."
        ),
    )
    trend_crawl_queries: str = Field(
        default="",
        description=(
            "Tavily 검색 쿼리 목록 (쉼표 구분). "
            "예: '2026 백엔드 채용 트렌드,AI 엔지니어 채용 공고'. ADR-101."
        ),
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
        description="ChromaDB persistence directory. 로컬 기본값 ./chroma_db. 배포 시 환경변수 CHROMA_PERSIST_DIR 로 주입.",
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
    redis_socket_timeout: float = Field(
        default=2.0,
        description="Redis 소켓 타임아웃 (초) — 장애 시 빠른 폴백을 위해 짧게 설정",
    )
    redis_connect_timeout: float = Field(
        default=2.0,
        description="Redis 연결 타임아웃 (초)",
    )
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Redis 타임아웃 시 자동 1회 재시도 여부 — ADR-126: ElastiCache 페일오버 중 순간적 TimeoutError 복구",
    )

    # ============================================
    # AWS / SageMaker Configuration (ADR-117, ADR-133)
    # ============================================
    aws_region: str = Field(
        default="ap-northeast-2",
        description="AWS 리전 (S3, SageMaker 공통)",
    )
    sagemaker_s3_bucket: str = Field(
        default="devths-storage-dev",
        description="SageMaker 학습 데이터 S3 버킷",
    )
    sagemaker_s3_training_prefix: str = Field(
        default="uploads/training-data/",
        description="학습 데이터 S3 경로 프리픽스",
    )
    sagemaker_pipeline_name: str = Field(
        default="exaone-finetuning-pipeline",
        description="SageMaker Pipeline 이름 (파인튜닝 파이프라인)",
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
    # Text Splitting / Chunking (ADR-060)
    # ============================================
    chunk_size: int = Field(
        default=2000,
        description="텍스트 청크 크기 (문자 기준, 약 500 토큰). Gemini-embedding-001 제한 2,048 토큰 이내",
    )
    chunk_overlap: int = Field(
        default=200,
        description="청크 간 오버랩 크기 (문자 기준, 약 50 토큰)",
    )
    # ADR-106 Phase 3: SemanticChunker (임베딩 유사도 기반 의미 분할)
    rag_use_semantic_chunker: bool = Field(
        default=False,
        description="Use SemanticChunker for embedding-based text splitting. ADR-106.",
    )
    semantic_chunker_threshold_type: str = Field(
        default="percentile",
        description="SemanticChunker breakpoint type: percentile, standard_deviation, interquartile. ADR-106.",
    )
    semantic_chunker_threshold_amount: float = Field(
        default=90.0,
        description="SemanticChunker threshold (percentile=90 → 상위 10%% 변화점에서 분할). ADR-106.",
    )
    # ADR-106 Phase 4: RAPTOR (재귀적 요약 트리)
    raptor_enabled: bool = Field(
        default=False,
        description="Enable RAPTOR tree construction on document upload. ADR-106.",
    )
    raptor_max_levels: int = Field(
        default=3,
        description="Maximum RAPTOR tree depth (levels of summarization). ADR-106.",
    )
    raptor_min_cluster_size: int = Field(
        default=3,
        description="Minimum texts in a cluster before stopping recursion. ADR-106.",
    )
    raptor_summary_max_docs: int = Field(
        default=5,
        description="Max documents per cluster to summarize (token limit). ADR-106.",
    )
    # ============================================
    # PDF Text Extraction (ADR-063)
    # ============================================
    pdf_extract_priority: str = Field(
        default="pdfplumber",
        description="PDF 텍스트 추출 우선 순위: 'pdfplumber' (디지털 PDF 우선) 또는 'ocr' (항상 OCR)",
    )

    # ============================================
    # RAG Configuration
    # ============================================
    rag_max_context_length: int = Field(
        default=6000,
        description="Maximum context length for RAG (characters). Phase 2: 4000→6000 (청크 도입으로 확장)",
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
        default="gemini-3.1-pro-preview",
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
        default=4096,
        description="면접 질문/평가 최대 토큰 수 (JSON 5개 질문 생성에 충분한 값)",
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
    def tavily_available(self) -> bool:
        """Tavily Search API 설정 여부. ADR-101/103."""
        return bool(self.tavily_api_key)

    @property
    def debate_available(self) -> bool:
        """Check if debate feature is available."""
        return self.eval_debate_enabled and self.openai_available


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
