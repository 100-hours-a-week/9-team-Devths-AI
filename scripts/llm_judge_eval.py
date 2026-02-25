#!/usr/bin/env python3
"""
ADR-091: LLM-as-Judge 평가 실행 스크립트.

4단계 파이프라인(langchain_only / rag / vllm / sagemaker)에 테스트셋을 넣고
Gemini Judge로 채점 후 결과를 Langfuse에 기록하고 JSON 파일로 저장한다.

사용 예::

    # RAG 단계만 평가
    python scripts/llm_judge_eval.py --stage rag

    # 4단계 전체 평가
    python scripts/llm_judge_eval.py --stage all

    # 테스트셋 경로 직접 지정
    python scripts/llm_judge_eval.py --stage rag --dataset data/eval/rag_test_dataset.json

    # 결과를 특정 파일에 저장
    python scripts/llm_judge_eval.py --stage rag --output data/eval/my_results.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가 (scripts/ 에서 app/ 임포트 가능하게)
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm_judge_eval")

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data" / "eval" / "rag_test_dataset.json"
RESULTS_DIR = _PROJECT_ROOT / "data" / "eval"

PIPELINE_STAGES = ["langchain_only", "rag", "vllm", "sagemaker"]


# ------------------------------------------------------------------
# 파이프라인 응답 생성 함수 (단계별)
# ------------------------------------------------------------------


async def _generate_langchain_only(user_message: str) -> str:
    """LangChain만 사용 (RAG 없이 Gemini 직접 호출) — 기준선(baseline)."""
    from app.config.settings import get_settings
    from app.services.llm_service import LLMService

    settings = get_settings()
    llm = LLMService(api_key=settings.google_api_key)

    chunks: list[str] = []
    async for chunk in llm.generate_response(
        user_message=user_message,
        context=None,
        history=[],
        system_prompt="당신은 취업·채용 전문 AI 어시스턴트입니다.",
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def _generate_rag(user_message: str) -> str:
    """RAG 파이프라인 (VectorDB 컨텍스트 + Gemini)."""
    from app.config.dependencies import get_services

    rag = get_services()
    # RAG 컨텍스트 검색 (user_id 없이 실행)
    context = await rag.retrieve_context(
        user_message=user_message,
        user_id="eval_script",
        conversation_types=["resume", "job_posting", "trend_data"],
    )

    chunks: list[str] = []
    async for chunk in rag.llm.generate_response(
        user_message=user_message,
        context=context or None,
        history=[],
        system_prompt="당신은 취업·채용 전문 AI 어시스턴트입니다.",
        user_id="eval_script",
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def _generate_vllm(user_message: str) -> str:
    """vLLM 서빙 파이프라인 (ExaONE 8B 등)."""
    from app.config.dependencies import get_services

    rag = get_services()
    if not rag.vllm:
        logger.warning("vLLM 서비스 미설정 — langchain_only로 대체")
        return await _generate_langchain_only(user_message)

    chunks: list[str] = []
    async for chunk in rag.vllm.generate_response(
        user_message=user_message,
        context=None,
        history=[],
        system_prompt="당신은 취업·채용 전문 AI 어시스턴트입니다.",
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def _generate_sagemaker(user_message: str) -> str:
    """SageMaker 학습 모델 (파인튜닝 모델 Endpoint 호출).

    SageMaker Endpoint URL은 SAGEMAKER_ENDPOINT_URL 환경변수로 지정.
    미설정 시 vllm 단계로 대체.
    """
    import aiohttp

    endpoint_url = os.getenv("SAGEMAKER_ENDPOINT_URL")
    if not endpoint_url:
        logger.warning("SAGEMAKER_ENDPOINT_URL 미설정 — vllm으로 대체")
        return await _generate_vllm(user_message)

    payload = {
        "inputs": user_message,
        "parameters": {"max_new_tokens": 512, "temperature": 0.7},
    }
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(endpoint_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp,
        ):
                resp.raise_for_status()
                data = await resp.json()
                # SageMaker 응답 형식: [{"generated_text": "..."}] 또는 {"generated_text": "..."}
                if isinstance(data, list):
                    return data[0].get("generated_text", "")
                return data.get("generated_text", str(data))
    except Exception as e:
        logger.error("SageMaker 호출 실패: %s — vllm으로 대체", e)
        return await _generate_vllm(user_message)


_STAGE_GENERATORS = {
    "langchain_only": _generate_langchain_only,
    "rag": _generate_rag,
    "vllm": _generate_vllm,
    "sagemaker": _generate_sagemaker,
}


# ------------------------------------------------------------------
# 평가 실행
# ------------------------------------------------------------------


async def evaluate_stage(
    stage: str,
    test_cases: list[dict[str, Any]],
    judge,
) -> list[dict[str, Any]]:
    """단일 파이프라인 단계에 대해 전체 테스트셋 평가 수행.

    Returns:
        각 테스트 케이스의 결과 dict 목록
    """
    generator = _STAGE_GENERATORS[stage]
    results: list[dict[str, Any]] = []

    for tc in test_cases:
        tc_id = tc["id"]
        question = tc["user_message"]
        reference = tc.get("reference_answer")

        logger.info("[%s] %s 평가 중...", stage, tc_id)

        try:
            # 파이프라인 추론
            answer = await generator(question)

            # Judge 채점
            judge_result = await judge.score(
                question=question,
                answer=answer,
                pipeline_stage=stage,
                reference_answer=reference,
                user_id="eval_script",
            )

            result = {
                "id": tc_id,
                "category": tc.get("category"),
                "pipeline_stage": stage,
                "question": question,
                "answer": answer,
                "reference_answer": reference,
                "scores": {s.name: s.score for s in judge_result.scores},
                "score_reasoning": {s.name: s.reasoning for s in judge_result.scores},
                "overall_score": judge_result.overall_score,
                "overall_reasoning": judge_result.overall_reasoning,
                "judge_model": judge_result.judge_model,
            }
            results.append(result)
            logger.info("[%s] %s 완료 — overall=%.1f", stage, tc_id, judge_result.overall_score)

        except Exception as e:
            logger.error("[%s] %s 평가 실패: %s", stage, tc_id, e)
            results.append(
                {
                    "id": tc_id,
                    "category": tc.get("category"),
                    "pipeline_stage": stage,
                    "question": question,
                    "answer": "",
                    "error": str(e),
                    "overall_score": None,
                }
            )

    return results


def _print_comparison_table(all_results: list[dict[str, Any]]) -> None:
    """4단계 점수 비교 테이블을 콘솔에 출력."""
    from collections import defaultdict

    stage_scores: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        if r.get("overall_score") is not None:
            stage_scores[r["pipeline_stage"]].append(r["overall_score"])

    print("\n" + "=" * 60)
    print("  ADR-091 LLM-as-Judge 결과 요약")
    print("=" * 60)
    print(f"{'파이프라인 단계':<20} {'평균 점수':>10} {'케이스 수':>10}")
    print("-" * 60)
    for stage in PIPELINE_STAGES:
        scores = stage_scores.get(stage, [])
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{stage:<20} {avg:>10.2f} {len(scores):>10}")
        else:
            print(f"{stage:<20} {'N/A':>10} {'0':>10}")
    print("=" * 60 + "\n")


async def main(args: argparse.Namespace) -> None:
    """메인 평가 루프."""
    from app.services.judge_service import JudgeService

    # 테스트셋 로드
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("테스트셋 파일 없음: %s", dataset_path)
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        test_cases: list[dict[str, Any]] = json.load(f)

    logger.info("테스트셋 로드 완료: %d개 케이스", len(test_cases))

    # 평가할 단계 결정
    stages = PIPELINE_STAGES if args.stage == "all" else [args.stage]
    logger.info("평가 단계: %s", stages)

    # Judge 초기화
    judge = JudgeService()

    # 단계별 평가
    all_results: list[dict[str, Any]] = []
    for stage in stages:
        logger.info("=== 단계 시작: %s ===", stage)
        stage_results = await evaluate_stage(stage, test_cases, judge)
        all_results.extend(stage_results)

    # 결과 저장
    output_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": {
            "evaluated_at": datetime.now().isoformat(),
            "stages": stages,
            "total_cases": len(all_results),
            "dataset": str(dataset_path),
        },
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("결과 저장 완료: %s", output_path)

    # 콘솔 비교 테이블 출력
    _print_comparison_table(all_results)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADR-091: LLM-as-Judge 파이프라인 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=[*PIPELINE_STAGES, "all"],
        default="rag",
        help="평가할 파이프라인 단계 (기본: rag)",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="테스트셋 JSON 경로 (기본: data/eval/rag_test_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 저장 경로 (기본: data/eval/results_YYYYMMDD_HHMMSS.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
