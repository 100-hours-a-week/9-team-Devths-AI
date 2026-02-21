"""
면접 데이터셋을 VectorDB에 임베딩하는 스크립트.

환경 변수:
- CHROMA_SERVER_HOST: Chroma 서버 호스트 (설정 시 서버 모드)
- CHROMA_SERVER_PORT: Chroma 서버 포트 (기본 8000)
- GOOGLE_API_KEY / GEMINI_API_KEY: 임베딩용 API 키
- INTERVIEW_DATASET_FILE: 데이터 파일 경로 (기본 data/interview_dataset_valid.json)

실행 예:
  poetry run python scripts/embed_interview_dataset.py
  INTERVIEW_DATASET_FILE=data/interview_dataset_train.json poetry run python scripts/embed_interview_dataset.py
  CHROMA_SERVER_HOST=vectordb.example.com poetry run python scripts/embed_interview_dataset.py
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

# .env 파일 로드 (경로 설정 후 import)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.services.vectordb_service import VectorDBService  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="면접 데이터셋을 VectorDB에 임베딩")
    parser.add_argument(
        "--file",
        "-f",
        default=os.getenv("INTERVIEW_DATASET_FILE", ""),
        help="데이터 파일 경로 (기본: data/interview_dataset_valid.json)",
    )
    return parser.parse_args()


async def main():
    """면접 데이터셋을 VectorDB에 저장"""
    args = parse_args()

    data_dir = PROJECT_ROOT / "data"
    data_file = Path(args.file) if args.file else data_dir / "interview_dataset_valid.json"
    if not data_file.is_absolute():
        data_file = PROJECT_ROOT / data_file
    if not data_file.exists():
        print(f"❌ 파일 없음: {data_file}", flush=True)
        sys.exit(1)

    print("🚀 면접 데이터셋 VectorDB 임베딩 시작...", flush=True)

    # 환경 변수로 Chroma/API 설정
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
    print("VectorDB 서비스 초기화 완료", flush=True)

    # JSONL 형식 (한 줄에 하나의 JSON 객체)
    print(f"데이터 파일 로드 중: {data_file}", flush=True)
    data = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"📊 총 {len(data)}개의 면접 Q&A 로드", flush=True)

    collection_name = "interview_feedback"
    batch_size = 100
    total_added = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        texts = []
        metadatas = []
        ids = []

        for j, item in enumerate(batch):
            text = f"질문: {item['question']}\n답변: {item['answer']}"
            texts.append(text)
            metadatas.append(
                {
                    "interview_type": item.get(
                        "interview_type", item.get("interviewType", "personality")
                    ),
                    "occupation": item.get("occupation", ""),
                    "experience": item.get("experience", ""),
                    "age_range": item.get("ageRange", ""),
                    "question_only": item["question"],
                    "answer_only": item["answer"],
                }
            )
            ids.append(f"interview_{i + j}")

        await vdb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            user_id=0,
            collection_name=collection_name,
        )

        total_added += len(batch)
        print(f"✅ {total_added}/{len(data)} 임베딩 완료...", flush=True)

    print(f"\n🎉 총 {total_added}개의 면접 Q&A가 VectorDB에 저장되었습니다!", flush=True)
    print(f"컬렉션: {collection_name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
