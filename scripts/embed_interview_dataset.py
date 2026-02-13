"""
면접 데이터셋을 VectorDB에 임베딩하는 스크립트
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv  # noqa: E402

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.services.vectordb_service import VectorDBService  # noqa: E402


async def main():
    """면접 데이터셋을 VectorDB에 저장"""
    print("🚀 면접 데이터셋 VectorDB 임베딩 시작...", flush=True)

    # VectorDB 서비스 초기화
    print("VectorDB 서비스 초기화 중...", flush=True)
    vdb = VectorDBService()
    print("VectorDB 서비스 초기화 완료", flush=True)

    # 데이터 로드 (valid 파일 사용 - 9.5MB, 빠른 배포용)
    data_dir = PROJECT_ROOT / "data"
    valid_file = data_dir / "interview_dataset_valid.json"

    # JSONL 형식 (한 줄에 하나의 JSON 객체)
    print(f"데이터 파일 로드 중: {valid_file}", flush=True)
    data = []
    with open(valid_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"📊 총 {len(data)}개의 면접 Q&A 로드", flush=True)

    # 컬렉션 이름: 문서 설계 interview_feedback (A안)
    collection_name = "interview_feedback"

    # 데이터 임베딩 (100개씩 배치)
    batch_size = 100
    total_added = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        texts = []
        metadatas = []
        ids = []

        for j, item in enumerate(batch):
            # 텍스트: 질문 + 답변 결합
            text = f"질문: {item['question']}\n답변: {item['answer']}"
            texts.append(text)

            # 메타데이터 (문서 A안: interview_type 필수 - technical | personality)
            metadatas.append(
                {
                    "interview_type": item.get(
                        "interview_type", item.get("interviewType", "technical")
                    ),
                    "occupation": item.get("occupation", ""),
                    "experience": item.get("experience", ""),
                    "age_range": item.get("ageRange", ""),
                    "question_only": item["question"],
                    "answer_only": item["answer"],
                }
            )

            # ID
            ids.append(f"interview_{i + j}")

        # VectorDB에 추가 (user_id=0은 공용 데이터)
        await vdb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            user_id=0,  # 공용 데이터
            collection_name=collection_name,
        )

        total_added += len(batch)
        print(f"✅ {total_added}/{len(data)} 임베딩 완료...", flush=True)

    print(f"\n🎉 총 {total_added}개의 면접 Q&A가 VectorDB에 저장되었습니다!", flush=True)
    print(f"컬렉션: {collection_name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
