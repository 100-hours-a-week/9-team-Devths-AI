"""
SageMaker Training Data Sync Tasks — ADR-117, ADR-114.

Celery 태스크: VectorDB(ChromaDB) 학습 데이터 → S3 업로드 + SageMaker Pipeline 트리거.
Redis를 통한 분산 태스크 처리.

- sync_training_data_task: VectorDB 컬렉션 데이터 추출 → 익명화 → S3 업로드
- trigger_sagemaker_pipeline_task: S3 업로드 완료 후 SageMaker Pipeline 실행

데이터 흐름 (ADR-117):
  Prod VectorDB(ChromaDB) → 익명화 → S3 스테이징 버킷 → SageMaker Pipeline 트리거
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime

import boto3

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# S3 설정
S3_BUCKET = os.getenv("SAGEMAKER_S3_BUCKET", "devths-storage-dev")
S3_TRAINING_DATA_PREFIX = os.getenv("SAGEMAKER_S3_TRAINING_PREFIX", "uploads/training-data/")

# SageMaker Pipeline 설정
SAGEMAKER_PIPELINE_NAME = os.getenv("SAGEMAKER_PIPELINE_NAME", "exaone-finetuning-pipeline")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# 학습에 사용할 VectorDB 컬렉션 목록
TRAINING_COLLECTIONS = [
    "trend_data",
    "interview_feedback",
]


def _anonymize_text(text: str) -> str:
    """간단한 익명화 처리 — 개인정보(이메일, 전화번호 등) 마스킹.

    TODO: 고도화 시 PII 탐지 모델 연동 (ADR-102 마스킹 태스크 참조)
    """

    # 이메일 마스킹
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "[EMAIL_MASKED]",
        text,
    )
    # 전화번호 마스킹 (한국 형식)
    text = re.sub(
        r"01[0-9]-?\d{3,4}-?\d{4}",
        "[PHONE_MASKED]",
        text,
    )
    # 주민등록번호 마스킹
    text = re.sub(
        r"\d{6}-?[1-4]\d{6}",
        "[RRN_MASKED]",
        text,
    )
    return text


def _extract_collection_data(vectordb, collection_type: str) -> list[dict]:
    """VectorDB 컬렉션에서 전체 데이터를 추출하여 학습용 포맷으로 변환."""
    collection = vectordb._get_collection(collection_type)
    result = collection.get(include=["documents", "metadatas"])

    records = []
    if not result or not result.get("documents"):
        logger.warning("[SageMakerSync] %s 컬렉션에 데이터가 없습니다.", collection_type)
        return records

    for doc_id, doc_text, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"] or [{}] * len(result["documents"]),
        strict=False,
    ):
        if not doc_text or len(doc_text.strip()) < 10:
            continue

        # 익명화 처리
        anonymized_text = _anonymize_text(doc_text)

        records.append(
            {
                "id": doc_id,
                "text": anonymized_text,
                "collection": collection_type,
                "metadata": {
                    k: v
                    for k, v in (metadata or {}).items()
                    if k not in ("user_id", "email", "name", "phone")
                },
            }
        )

    return records


def _upload_to_s3(records: list[dict], collection_type: str, timestamp: str) -> str:
    """학습 데이터를 JSONL 형식으로 S3에 업로드."""
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    s3_key = f"{S3_TRAINING_DATA_PREFIX}{timestamp}/{collection_type}.jsonl"

    jsonl_content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    logger.info("[SageMakerSync] S3 업로드 완료: %s (%d건)", s3_uri, len(records))
    return s3_uri


@celery_app.task(bind=True, name="app.tasks.sagemaker_sync_tasks.sync_training_data_task")
def sync_training_data_task(self):
    """VectorDB 학습 데이터 → S3 업로드 태스크 (Celery Beat 스케줄러용).

    ADR-117 데이터 흐름:
      Prod VectorDB(ChromaDB) 익명화 데이터 Pull → S3 스테이징 버킷 업로드

    학습 대상 컬렉션: trend_data, interview_feedback
    """
    logger.info("[SageMakerSync] 학습 데이터 S3 동기화 시작")

    try:
        result = asyncio.run(_sync_training_data_async())
        logger.info(
            "[SageMakerSync] 동기화 완료: 총 %d건 업로드",
            result.get("total_records", 0),
        )
        return result

    except Exception as e:
        logger.error("[SageMakerSync] 학습 데이터 동기화 실패: %s", e)
        raise self.retry(exc=e, countdown=600, max_retries=3) from e


async def _sync_training_data_async() -> dict:
    """비동기 학습 데이터 추출 및 S3 업로드 실행."""
    from app.config.settings import get_settings

    settings = get_settings()

    # VectorDB 연결 (기존 인프라 재사용)
    from app.services.vectordb_service import VectorDBService

    vectordb = VectorDBService(
        api_key=settings.google_api_key,
        persist_directory=settings.chroma_persist_dir,
        chroma_server_host=settings.chroma_server_host,
        chroma_server_port=settings.chroma_server_port,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_records = 0
    uploaded_files = []

    for collection_type in TRAINING_COLLECTIONS:
        logger.info("[SageMakerSync] %s 컬렉션 데이터 추출 중...", collection_type)

        # 데이터 추출 + 익명화
        records = _extract_collection_data(vectordb, collection_type)

        if not records:
            logger.info("[SageMakerSync] %s 컬렉션: 데이터 없음, 스킵", collection_type)
            continue

        # S3 업로드
        s3_uri = _upload_to_s3(records, collection_type, timestamp)
        total_records += len(records)
        uploaded_files.append(
            {
                "collection": collection_type,
                "s3_uri": s3_uri,
                "record_count": len(records),
            }
        )

    return {
        "status": "completed",
        "timestamp": timestamp,
        "total_records": total_records,
        "s3_base_path": f"s3://{S3_BUCKET}/{S3_TRAINING_DATA_PREFIX}{timestamp}/",
        "uploaded_files": uploaded_files,
    }


@celery_app.task(
    bind=True,
    name="app.tasks.sagemaker_sync_tasks.trigger_sagemaker_pipeline_task",
)
def trigger_sagemaker_pipeline_task(self, s3_data_path: str = ""):
    """S3 업로드 완료 후 SageMaker Pipeline 트리거.

    sync_training_data_task 완료 후 체이닝하여 호출하거나 수동 실행 가능.

    Args:
        s3_data_path: S3 학습 데이터 경로 (미지정 시 latest 사용)
    """
    logger.info("[SageMakerSync] SageMaker Pipeline 트리거: %s", SAGEMAKER_PIPELINE_NAME)

    try:
        sm_client = boto3.client("sagemaker", region_name=AWS_REGION)

        params = []
        if s3_data_path:
            params.append({"Name": "ChromaDataS3Uri", "Value": s3_data_path})

        response = sm_client.start_pipeline_execution(
            PipelineName=SAGEMAKER_PIPELINE_NAME,
            PipelineParameters=params,
            PipelineExecutionDescription=f"Auto-triggered by Celery Beat at {datetime.now().isoformat()}",
        )

        execution_arn = response.get("PipelineExecutionArn", "")
        logger.info("[SageMakerSync] Pipeline 실행 시작: %s", execution_arn)

        return {
            "status": "pipeline_triggered",
            "pipeline_name": SAGEMAKER_PIPELINE_NAME,
            "execution_arn": execution_arn,
            "s3_data_path": s3_data_path,
        }

    except Exception as e:
        logger.error("[SageMakerSync] Pipeline 트리거 실패: %s", e)
        raise self.retry(exc=e, countdown=300, max_retries=2) from e
