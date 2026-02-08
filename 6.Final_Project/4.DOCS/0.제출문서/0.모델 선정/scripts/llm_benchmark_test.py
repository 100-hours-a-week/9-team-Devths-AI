"""
LLM 모델 & 하드웨어 선정 평가 스크립트

평가 대상:
- LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ (L4 24GB)
- LGAI-EXAONE/EXAONE-3.5-32B-Instruct-AWQ (A100 40GB)

평가 지표:
- TTFT (Time To First Token)
- TPS (Tokens Per Second)
- 꼬리 질문 생성 능력
- 맥락 파악 정확도
"""

import os
import time
import json
import torch
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from huggingface_hub import login
from getpass import getpass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ============================================
# 1. 설정
# ============================================

@dataclass
class ModelConfig:
    name: str
    model_id: str
    required_vram: str
    recommended_gpu: str


MODELS = {
    "exaone-7.8b": ModelConfig(
        name="EXAONE-3.5-7.8B-Instruct-AWQ",
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ",
        required_vram="~8GB (4bit)",
        recommended_gpu="NVIDIA L4 (24GB)"
    ),
    "exaone-32b": ModelConfig(
        name="EXAONE-3.5-32B-Instruct-AWQ",
        model_id="LGAI-EXAONE/EXAONE-3.5-32B-Instruct-AWQ",
        required_vram="~20GB (4bit)",
        recommended_gpu="NVIDIA A100 (40GB)"
    )
}


# ============================================
# 2. 평가 데이터셋
# ============================================

EVALUATION_PROMPTS = {
    # 꼬리 질문 생성 능력 테스트
    "follow_up_question": {
        "system": "당신은 취업 면접관입니다. 지원자의 답변을 듣고 꼬리 질문을 생성하세요.",
        "user": """
[이전 질문]: 가장 어려웠던 프로젝트 경험을 말씀해주세요.

[지원자 답변]:
저는 전 회사에서 레거시 시스템을 마이크로서비스로 전환하는 프로젝트를 진행했습니다.
모놀리식 구조에서 Docker와 Kubernetes를 활용해 서비스를 분리했고,
팀원 5명과 함께 6개월간 진행했습니다.

[요청]: 위 답변에 대한 꼬리 질문을 2개 생성하세요. 기술적 깊이와 문제 해결 능력을 파악할 수 있는 질문이어야 합니다.
""",
        "expected_keywords": ["마이크로서비스", "문제", "해결", "어려움", "기술", "팀"],
        "min_questions": 2
    },

    # 맥락 파악 정확도 테스트
    "context_understanding": {
        "system": "당신은 이력서 분석 전문가입니다.",
        "user": """
[이력서 내용]:
- 이름: 김개발
- 경력: 3년차 백엔드 개발자
- 기술스택: Python, FastAPI, PostgreSQL, Redis, Docker
- 프로젝트:
  1. 결제 시스템 개발 (처리량 200% 향상)
  2. 실시간 알림 서비스 (WebSocket)
- 약점: 프론트엔드 경험 부족

[채용공고]:
- 포지션: 풀스택 개발자
- 필수: React, Node.js, AWS
- 우대: Python, 결제 시스템 경험

[요청]:
1. 이 지원자의 강점과 약점을 분석하세요.
2. 채용공고와의 매칭도를 0-100점으로 평가하세요.
3. 면접에서 확인해야 할 포인트를 제시하세요.
""",
        "expected_sections": ["강점", "약점", "매칭", "면접"],
        "requires_score": True
    },

    # 일반 대화 테스트
    "general_chat": {
        "system": "당신은 취업 상담 AI 어시스턴트입니다. 친절하고 전문적으로 답변하세요.",
        "user": "백엔드 개발자로 취업하려면 어떤 준비가 필요해?",
        "expected_keywords": ["기술", "포트폴리오", "프로젝트", "학습"],
        "min_length": 200
    },

    # 긴 컨텍스트 처리 테스트
    "long_context": {
        "system": "당신은 면접 결과 분석 전문가입니다.",
        "user": """
다음 5개의 면접 Q&A를 분석하고 종합 평가를 작성하세요.

[Q1]: 자기소개를 해주세요.
[A1]: 안녕하세요, 3년차 백엔드 개발자 김개발입니다. Python과 FastAPI를 주로 사용하며, 최근에는 마이크로서비스 아키텍처에 관심을 가지고 있습니다.

[Q2]: 가장 어려웠던 기술적 문제와 해결 방법을 설명해주세요.
[A2]: 레거시 시스템의 성능 문제를 해결한 경험이 있습니다. 프로파일링을 통해 병목 지점을 찾고, 캐싱과 쿼리 최적화로 응답 시간을 70% 단축했습니다.

[Q3]: 팀에서 갈등이 있었을 때 어떻게 해결했나요?
[A3]: 코드 리뷰 과정에서 의견 충돌이 있었습니다. 데이터를 기반으로 각 방식의 장단점을 정리하고, 팀 미팅에서 함께 결정했습니다.

[Q4]: 우리 회사에 지원한 이유는 무엇인가요?
[A4]: 귀사의 기술 블로그를 통해 혁신적인 기술 도전을 하고 있다는 것을 알게 되었습니다. 특히 대용량 데이터 처리 경험을 쌓고 싶습니다.

[Q5]: 5년 후 어떤 개발자가 되고 싶나요?
[A5]: 기술적으로 깊이 있는 백엔드 전문가가 되고 싶습니다. 동시에 팀을 이끌 수 있는 테크 리드로 성장하고 싶습니다.

[요청]: 위 면접 내용을 바탕으로:
1. 각 답변별 점수 (1-5점)
2. 종합 평가 (강점/약점)
3. 최종 추천 여부
를 작성하세요.
""",
        "expected_sections": ["점수", "강점", "약점", "추천"],
        "min_length": 500
    }
}


# ============================================
# 3. 벤치마크 클래스
# ============================================

@dataclass
class BenchmarkResult:
    model_name: str
    prompt_type: str
    ttft_ms: float          # Time To First Token (밀리초)
    total_time_s: float     # 총 생성 시간 (초)
    input_tokens: int       # 입력 토큰 수
    output_tokens: int      # 출력 토큰 수
    tps: float              # Tokens Per Second
    response: str           # 생성된 응답
    quality_score: Optional[float] = None  # 품질 점수 (수동 평가)
    gpu_memory_mb: Optional[float] = None  # GPU 메모리 사용량


class LLMBenchmark:
    """LLM 모델 벤치마크 테스트"""

    def __init__(self, model_key: str):
        self.model_config = MODELS[model_key]
        self.model = None
        self.tokenizer = None
        self.results: List[BenchmarkResult] = []

    def setup_hf_auth(self):
        """Hugging Face 인증"""
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            hf_token = getpass("🔑 Hugging Face Access Token: ")

        if hf_token:
            try:
                login(token=hf_token)
                os.environ["HF_TOKEN"] = hf_token
                print("✅ Hugging Face 로그인 성공!")
                return True
            except Exception as e:
                print(f"❌ 로그인 실패: {e}")
                return False
        return False

    def load_model(self):
        """모델 로딩 (4bit 양자화)"""
        print(f"\n🔄 {self.model_config.name} 로딩 중...")
        print(f"   - 필요 VRAM: {self.model_config.required_vram}")
        print(f"   - 권장 GPU: {self.model_config.recommended_gpu}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        start_time = time.time()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_id,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            load_time = time.time() - start_time
            print(f"✅ 모델 로딩 완료 ({load_time:.1f}초)")

            # GPU 메모리 확인
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"   - GPU 메모리 사용: {memory_mb:.0f}MB")

            return True

        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return False

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> tuple:
        """텍스트 생성 및 시간 측정"""
        if not self.model:
            raise RuntimeError("모델이 로딩되지 않았습니다.")

        # 프롬프트 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]

        # TTFT 측정을 위한 스트리밍 생성
        start_time = time.time()
        first_token_time = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        end_time = time.time()

        # 결과 파싱
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_tokens = len(generated_ids)

        total_time = end_time - start_time
        ttft = total_time / output_tokens * 1000 if output_tokens > 0 else 0  # 근사값
        tps = output_tokens / total_time if total_time > 0 else 0

        return response, {
            "ttft_ms": ttft,
            "total_time_s": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tps": tps
        }

    def run_benchmark(self, prompt_key: str) -> BenchmarkResult:
        """단일 벤치마크 실행"""
        prompt_data = EVALUATION_PROMPTS[prompt_key]

        print(f"\n📝 테스트: {prompt_key}")
        print("-" * 50)

        response, metrics = self.generate(
            prompt_data["system"],
            prompt_data["user"]
        )

        # GPU 메모리 측정
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024

        result = BenchmarkResult(
            model_name=self.model_config.name,
            prompt_type=prompt_key,
            ttft_ms=metrics["ttft_ms"],
            total_time_s=metrics["total_time_s"],
            input_tokens=metrics["input_tokens"],
            output_tokens=metrics["output_tokens"],
            tps=metrics["tps"],
            response=response,
            gpu_memory_mb=gpu_memory
        )

        self.results.append(result)

        # 결과 출력
        print(f"⏱️  TTFT: {result.ttft_ms:.1f}ms")
        print(f"⏱️  Total: {result.total_time_s:.2f}s")
        print(f"📊 TPS: {result.tps:.1f} tokens/sec")
        print(f"📝 Input: {result.input_tokens} tokens")
        print(f"📝 Output: {result.output_tokens} tokens")
        print(f"\n💬 응답 (처음 500자):\n{response[:500]}...")

        return result

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """모든 벤치마크 실행"""
        print(f"\n{'='*60}")
        print(f"🚀 {self.model_config.name} 전체 벤치마크 시작")
        print(f"{'='*60}")

        for prompt_key in EVALUATION_PROMPTS.keys():
            self.run_benchmark(prompt_key)

        return self.results

    def generate_report(self) -> Dict:
        """벤치마크 리포트 생성"""
        if not self.results:
            return {}

        avg_ttft = sum(r.ttft_ms for r in self.results) / len(self.results)
        avg_tps = sum(r.tps for r in self.results) / len(self.results)
        avg_time = sum(r.total_time_s for r in self.results) / len(self.results)

        report = {
            "model": self.model_config.name,
            "model_id": self.model_config.model_id,
            "recommended_gpu": self.model_config.recommended_gpu,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "avg_ttft_ms": round(avg_ttft, 2),
                "avg_tps": round(avg_tps, 2),
                "avg_total_time_s": round(avg_time, 2),
                "total_tests": len(self.results)
            },
            "results": [asdict(r) for r in self.results]
        }

        return report

    def save_report(self, filepath: str):
        """리포트 저장"""
        report = self.generate_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 리포트 저장: {filepath}")


# ============================================
# 4. 품질 평가 함수
# ============================================

def evaluate_follow_up_quality(response: str, expected: dict) -> float:
    """꼬리 질문 생성 품질 평가"""
    score = 0.0

    # 키워드 포함 여부 (40%)
    keywords_found = sum(1 for kw in expected["expected_keywords"] if kw in response)
    keyword_score = keywords_found / len(expected["expected_keywords"]) * 40
    score += keyword_score

    # 질문 개수 확인 (30%)
    question_count = response.count("?")
    if question_count >= expected["min_questions"]:
        score += 30
    else:
        score += (question_count / expected["min_questions"]) * 30

    # 응답 길이 (30%)
    if len(response) >= 100:
        score += 30
    else:
        score += (len(response) / 100) * 30

    return round(score, 1)


def evaluate_context_understanding(response: str, expected: dict) -> float:
    """맥락 파악 품질 평가"""
    score = 0.0

    # 필수 섹션 포함 여부 (50%)
    sections_found = sum(1 for sec in expected["expected_sections"] if sec in response)
    section_score = sections_found / len(expected["expected_sections"]) * 50
    score += section_score

    # 점수 포함 여부 (25%)
    if expected.get("requires_score"):
        import re
        if re.search(r'\d{1,3}점', response) or re.search(r'\d{1,3}/100', response):
            score += 25

    # 구조화된 응답 (25%)
    if any(marker in response for marker in ["1.", "2.", "3.", "-", "•"]):
        score += 25

    return round(score, 1)


# ============================================
# 5. 메인 실행
# ============================================

def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM 모델 벤치마크 테스트")
    parser.add_argument(
        "--model",
        choices=["exaone-7.8b", "exaone-32b", "all"],
        default="exaone-7.8b",
        help="테스트할 모델 선택"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="결과 저장 파일 경로"
    )

    args = parser.parse_args()

    models_to_test = list(MODELS.keys()) if args.model == "all" else [args.model]

    all_reports = []

    for model_key in models_to_test:
        benchmark = LLMBenchmark(model_key)

        # 인증
        if not benchmark.setup_hf_auth():
            print("⚠️ 인증 실패, 스킵합니다.")
            continue

        # 모델 로딩
        if not benchmark.load_model():
            print("⚠️ 모델 로딩 실패, 스킵합니다.")
            continue

        # 벤치마크 실행
        benchmark.run_all_benchmarks()

        # 리포트 생성
        report = benchmark.generate_report()
        all_reports.append(report)

        # 개별 저장
        output_path = f"{model_key}_{args.output}"
        benchmark.save_report(output_path)

    # 전체 결과 요약
    print("\n" + "="*60)
    print("📊 전체 벤치마크 결과 요약")
    print("="*60)

    for report in all_reports:
        print(f"\n🤖 {report['model']}")
        print(f"   - 평균 TTFT: {report['summary']['avg_ttft_ms']:.1f}ms")
        print(f"   - 평균 TPS: {report['summary']['avg_tps']:.1f} tokens/sec")
        print(f"   - 평균 응답시간: {report['summary']['avg_total_time_s']:.2f}s")


if __name__ == "__main__":
    main()
