"""
캘린더 파싱 서비스 테스트 스크립트

실행 방법:
cd /Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model
python test_calendar_parsing.py
"""

import asyncio
import os

import pytest
from dotenv import load_dotenv

from app.services.calendar_parsing import CalendarParsingService

# .env 파일 로드
load_dotenv()


# [CI/CD] GitHub Actions 환경에서 API Key가 없을 경우 테스트를 건너뛰기 위한 설정
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
async def test_text_parsing():
    """텍스트 기반 채용공고 파싱 테스트"""
    print("=" * 80)
    print("테스트 1: 텍스트 기반 채용공고 파싱")
    print("=" * 80)

    sample_text = """
    [카카오 2026년 상반기 신입 개발자 채용]

    회사: 카카오
    직무: 백엔드 개발자 (Server)

    전형 일정:
    - 서류 접수 마감: 2026년 2월 15일 (토) 23:59
    - 코딩테스트: 2026년 2월 22일 (토) 14:00
    - 1차 면접: 2026년 3월 5일 ~ 3월 7일
    - 2차 면접 (최종): 2026년 3월 12일 ~ 3월 14일
    - 최종 합격 발표: 2026년 3월 20일

    자격 요건:
    - 2026년 2월 졸업 예정자 또는 졸업자
    - Java, Spring, MySQL 등 백엔드 기술 스택 경험자 우대
    - Docker, Kubernetes 경험자 우대

    #카카오 #백엔드 #신입 #Java #Spring
    """

    service = CalendarParsingService()
    result = await service.parse_from_text(sample_text)

    print("\n✅ 파싱 결과:")
    print(f"  회사명: {result.company}")
    print(f"  포지션: {result.position}")
    print(f"  해시태그: {', '.join(result.hashtags)}")
    print(f"\n  일정 ({len(result.schedules)}개):")
    for schedule in result.schedules:
        time_str = f" {schedule.time}" if schedule.time else ""
        print(f"    - {schedule.stage}: {schedule.date}{time_str}")

    return result


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
async def test_text_parsing_simple():
    """간단한 텍스트 파싱 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: 간단한 텍스트 파싱")
    print("=" * 80)

    sample_text = """
    네이버 프론트엔드 개발자 채용

    서류 마감: 2026-01-30
    코딩테스트: 2026-02-05 10:00
    면접: 2026-02-15
    """

    service = CalendarParsingService()
    result = await service.parse_from_text(sample_text)

    print("\n✅ 파싱 결과:")
    print(f"  회사명: {result.company}")
    print(f"  포지션: {result.position}")
    print(f"  해시태그: {', '.join(result.hashtags)}")
    print(f"\n  일정 ({len(result.schedules)}개):")
    for schedule in result.schedules:
        time_str = f" {schedule.time}" if schedule.time else ""
        print(f"    - {schedule.stage}: {schedule.date}{time_str}")

    return result


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
async def test_empty_text():
    """빈 텍스트 처리 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 정보가 부족한 텍스트")
    print("=" * 80)

    sample_text = "채용 공고입니다."

    service = CalendarParsingService()
    result = await service.parse_from_text(sample_text)

    print("\n✅ 파싱 결과 (정보 부족 시):")
    print(f"  회사명: {result.company}")
    print(f"  포지션: {result.position}")
    print(f"  해시태그: {', '.join(result.hashtags)}")
    print(f"  일정: {len(result.schedules)}개")

    return result


async def main():
    """메인 테스트 실행"""
    # GOOGLE_API_KEY 환경 변수 확인
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ 오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("다음 명령어로 설정하세요:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    print("\n🚀 캘린더 파싱 서비스 테스트 시작\n")

    try:
        # 테스트 1: 상세 텍스트 파싱
        await test_text_parsing()

        # 테스트 2: 간단한 텍스트 파싱
        await test_text_parsing_simple()

        # 테스트 3: 정보 부족 텍스트
        await test_empty_text()

        print("\n" + "=" * 80)
        print("✅ 모든 테스트 완료!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
