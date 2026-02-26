"""
자동 배포(CI/CD) 파이프라인에서 트리거되어 data/ 파일 변경 사항을 VectorDB에 동기화하는 래퍼 스크립트.
주요 로직은 기존 embed_interview_dataset.py 등을 재사용하여 확장 가능하게 설계합니다.
"""

import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def run_interview_dataset_embedding():
    """면접 데이터셋 자동 임베딩 실행"""
    from scripts.embed_interview_dataset import main as embed_main
    logger.info("Starting Interview Dataset Embedding process...")
    try:
        # embed_interview_dataset.py 내부의 argparse 충돌 방지를 위해 sys.argv 초기화
        old_argv = sys.argv
        sys.argv = ['embed_interview_dataset.py']
        await embed_main()
        sys.argv = old_argv
        logger.info("Successfully finished Interview Dataset Embedding.")
    except Exception as e:
        logger.error(f"Failed to embed interview dataset: {e}")
        raise

async def auto_embed_all_data():
    """모든 데이터 소스 동기화 매니저"""
    logger.info("========================================")
    logger.info("🚀 AI Service Auto Embedding Triggered!")
    logger.info("========================================")

    # 1. 면접 데이터셋 (필수)
    # 추후 추가될 데이터 컬렉션(e.g., 직무 공고 등)이 있다면 아래에 순차적으로 추가합니다.
    await run_interview_dataset_embedding()

    logger.info("✅ All auto-embedding tasks completed.")

if __name__ == "__main__":
    try:
        # 이 스크립트는 컨테이너 내부(docker exec)에서 실행됨.
        # Docker 컨테이너 내부는 이미 환경변수(GOOGLE_API_KEY 등)가 로드된 상태 모델임.
        asyncio.run(auto_embed_all_data())
    except Exception as e:
        logger.critical("Auto-embedding script failed!", exc_info=e)
        sys.exit(1)
