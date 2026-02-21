"""
과거 면접 결과를 VectorDB에 일괄 적재하는 CLI 스크립트.

입력 JSON 형식:
[
  {
    "session_id": "session_123",
    "user_id": "42",
    "room_id": "room_001",
    "interview_type": "technical",
    "qa_results": [
      {
        "question": "React의 Virtual DOM이 무엇인가요?",
        "answer": "실제 DOM과 비교해서 ...",
        "question_index": 0,
        "score": 4,
        "verdict": "적절",
        "reasoning": "핵심 개념을 잘 이해하고 있음",
        "recommended_answer": null
      }
    ]
  }
]

실행 예:
  poetry run python scripts/ingest_interview_results.py --file data/past_interviews.json
  CHROMA_SERVER_HOST=vectordb poetry run python scripts/ingest_interview_results.py -f data/past_interviews.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.schemas.ingestion import InterviewIngestionInput  # noqa: E402
from app.services.interview_ingestion_service import InterviewIngestionService  # noqa: E402
from app.services.vectordb_service import VectorDBService  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="과거 면접 결과를 VectorDB에 일괄 적재")
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="면접 결과 JSON 파일 경로",
    )
    return parser.parse_args()


async def main():
    """면접 결과 JSON → VectorDB 적재"""
    args = parse_args()

    data_file = Path(args.file)
    if not data_file.is_absolute():
        data_file = PROJECT_ROOT / data_file
    if not data_file.exists():
        print(f"❌ 파일 없음: {data_file}", flush=True)
        sys.exit(1)

    print("🚀 면접 결과 VectorDB 적재 시작...", flush=True)

    # VectorDB 초기화
    chroma_host = os.getenv("CHROMA_SERVER_HOST")
    chroma_port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    if chroma_host:
        print(f"   Chroma 서버 모드: {chroma_host}:{chroma_port}", flush=True)
    else:
        print(f"   Chroma embedded 모드: {persist_dir}", flush=True)

    vdb = VectorDBService(
        api_key=api_key,
        persist_directory=persist_dir,
        chroma_server_host=chroma_host,
        chroma_server_port=chroma_port,
    )
    ingestion_service = InterviewIngestionService(vdb)
    print("✅ 서비스 초기화 완료", flush=True)

    # JSON 로드
    print(f"📂 데이터 로드: {data_file}", flush=True)
    with open(data_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    # 단일 세션이면 리스트로 감싸기
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    print(f"📊 총 {len(raw_data)}개 세션 발견", flush=True)

    total_ingested = 0

    for i, session_data in enumerate(raw_data, 1):
        try:
            input_data = InterviewIngestionInput(**session_data)
            result = await ingestion_service.ingest_session(input_data)
            total_ingested += result["ingested_count"]
            print(
                f"  [{i}/{len(raw_data)}] session={input_data.session_id} "
                f"→ {result['ingested_count']}건 적재",
                flush=True,
            )
        except Exception as e:
            print(f"  [{i}/{len(raw_data)}] ❌ 오류: {e}", flush=True)

    print(f"\n🎉 총 {total_ingested}건 VectorDB 적재 완료!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
