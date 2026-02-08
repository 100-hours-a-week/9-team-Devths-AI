# LLM 모델 벤치마크 사용 가이드

## 📋 준비사항

### 1. 필요한 패키지 설치

```bash
pip install openai asyncio
```

### 2. vLLM 서버 실행

테스트할 모델의 vLLM 서버를 먼저 실행해야 합니다.

```bash
# 예: Llama-3-Korean-Bllossom-8B
docker run -d --gpus all \
  --name vllm-server \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

### 3. 서버 준비 확인

```bash
# 헬스 체크
curl http://localhost:8000/health

# 모델 정보 확인
curl http://localhost:8000/v1/models
```

---

## 🚀 벤치마크 실행

### 기본 사용법

```bash
python3 benchmark_llm_models.py \
  --model "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
  --runs 5
```

### 옵션 설명

| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--model` | 모델 이름 (필수) | - | `"MLP-KTLim/llama-3-Korean-Bllossom-8B"` |
| `--base-url` | vLLM 서버 URL | `http://localhost:8000/v1` | `http://10.0.0.1:8000/v1` |
| `--runs` | 각 테스트 반복 횟수 | 5 | 10 |
| `--output` | 결과 저장 파일 (JSON) | - | `results_llama3.json` |

### 전체 옵션 예시

```bash
python3 benchmark_llm_models.py \
  --model "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
  --base-url "http://localhost:8000/v1" \
  --runs 10 \
  --output "results_llama3_korean.json"
```

---

## 📊 테스트 항목

벤치마크는 다음 5가지 테스트를 실행합니다:

### 1. 분석 + 매칭도 (Analysis)
- **프롬프트**: 이력서와 채용공고 분석
- **Temperature**: 0.3
- **Max Tokens**: 2048
- **측정 지표**: 응답 시간, 토큰 수

### 2. 면접 질문 생성 (Interview Question)
- **프롬프트**: 기술 면접 질문 생성
- **Temperature**: 0.8
- **Max Tokens**: 512
- **측정 지표**: 응답 시간, 토큰 수

### 3. 대화 + RAG (Chat with RAG)
- **프롬프트**: RAG 컨텍스트 기반 대화
- **Temperature**: 0.7
- **Max Tokens**: 2048
- **측정 지표**: 응답 시간, TTFT, TPS

### 4. 한국어 이해도 (Korean Understanding)
- **프롬프트**: 한국어 문장 분석 및 요약
- **Temperature**: 0.3
- **Max Tokens**: 512
- **측정 지표**: 응답 시간, 정확도

### 5. 꼬리 질문 생성 (Follow-up Question)
- **프롬프트**: 이전 답변 기반 심화 질문
- **Temperature**: 0.8
- **Max Tokens**: 256
- **측정 지표**: 응답 시간, 연관성

---

## 📈 결과 확인

### 터미널 출력

실행 중 터미널에서 실시간으로 진행 상황을 확인할 수 있습니다:

```
============================================================
LLM 모델 벤치마크 시작
모델: MLP-KTLim/llama-3-Korean-Bllossom-8B
시간: 2026-01-20 15:30:00
============================================================

[테스트 1] 분석 + 매칭도 프롬프트 (5회 반복)
  실행 1/5... ✓ 3.45s
  실행 2/5... ✓ 3.52s
  실행 3/5... ✓ 3.48s
  실행 4/5... ✓ 3.51s
  실행 5/5... ✓ 3.49s

[테스트 2] 면접 질문 생성 프롬프트 (5회 반복)
  실행 1/5... ✓ 1.23s
  ...
```

### 결과 요약

테스트 완료 후 요약 정보가 출력됩니다:

```
============================================================
테스트 결과 요약
============================================================

[분석 + 매칭도]
  성공률: 100.0%
  평균 응답 시간: 3.490s
  평균 TPS: 32.5
  평균 토큰 수: 1250
  P50 Latency: 3.490s
  P95 Latency: 3.520s

[면접 질문 생성]
  성공률: 100.0%
  평균 응답 시간: 1.230s
  평균 TPS: 28.3
  평균 토큰 수: 180
  P50 Latency: 1.230s
  P95 Latency: 1.250s
  ...
```

### 마크다운 테이블

문서에 바로 복사할 수 있는 마크다운 테이블도 생성됩니다:

```markdown
| 테스트 항목 | 평균 응답 시간 | TTFT | TPS | 토큰 수 | P95 Latency |
|------------|:-------------:|:----:|:---:|:-------:|:-----------:|
| 분석 + 매칭도 | 3.49s | - | 32.5 | 1250 | 3.52s |
| 면접 질문 생성 | 1.23s | - | 28.3 | 180 | 1.25s |
| 대화 + RAG | 2.15s | 0.450s | 35.2 | 520 | 2.20s |
| 한국어 이해도 | 1.85s | - | 30.1 | 350 | 1.90s |
| 꼬리 질문 생성 | 0.95s | - | 29.8 | 150 | 1.00s |
```

### JSON 결과 파일

`--output` 옵션을 사용하면 상세한 결과가 JSON 파일로 저장됩니다:

```json
{
  "model_name": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
  "timestamp": "2026-01-20T15:30:00",
  "tests": {
    "analysis": {
      "test_name": "분석 + 매칭도",
      "success_rate": 100.0,
      "avg_total_time": 3.49,
      "avg_ttft": null,
      "avg_tps": 32.5,
      "avg_tokens": 1250,
      "p50_latency": 3.49,
      "p95_latency": 3.52,
      "p99_latency": 3.52,
      "sample_response": "..."
    },
    ...
  }
}
```

---

## 🔄 여러 모델 테스트 자동화

모든 모델을 순차적으로 테스트하는 스크립트:

```bash
#!/bin/bash

# 테스트할 모델 목록
MODELS=(
  "MLP-KTLim/llama-3-Korean-Bllossom-8B"
  "Qwen/Qwen3-8B"
  "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  "skt/A.X-3.1-Light"
)

# 각 모델별로 테스트
for model in "${MODELS[@]}"; do
  echo "=========================================="
  echo "테스트 시작: $model"
  echo "=========================================="
  
  # 모델 이름에서 파일명 생성 (슬래시를 언더스코어로 변경)
  filename=$(echo "$model" | tr '/' '_')
  
  # 벤치마크 실행
  python3 benchmark_llm_models.py \
    --model "$model" \
    --runs 5 \
    --output "results_${filename}.json"
  
  echo ""
  echo "결과 저장: results_${filename}.json"
  echo ""
  
  # 다음 모델 테스트 전 대기 (서버 안정화)
  sleep 10
done

echo "모든 테스트 완료!"
```

---

## 📝 결과 분석 팁

### 1. 성능 지표 해석

- **평균 응답 시간**: 전체 요청-응답 시간 (낮을수록 좋음)
- **TTFT (Time to First Token)**: 첫 토큰까지 시간 (낮을수록 좋음, 스트리밍 시)
- **TPS (Tokens Per Second)**: 초당 토큰 생성 속도 (높을수록 좋음)
- **P95 Latency**: 95%의 요청이 이 시간 내에 완료 (안정성 지표)

### 2. 모델 비교 시 고려사항

- 동일한 하드웨어 환경에서 테스트
- 동일한 `--runs` 값 사용 (최소 5회 이상 권장)
- GPU 온도가 안정화된 후 테스트 (워밍업)
- 다른 프로세스가 GPU를 사용하지 않는지 확인

### 3. 이상치 처리

- P95, P99 값이 평균보다 크게 높으면 불안정한 성능
- 성공률이 100%가 아니면 에러 로그 확인 필요
- 첫 번째 실행은 모델 로딩 시간이 포함될 수 있음

---

## 🐛 문제 해결

### vLLM 서버가 응답하지 않음

```bash
# 서버 상태 확인
docker logs vllm-server --tail 50

# 서버 재시작
docker restart vllm-server

# 포트 확인
netstat -tlnp | grep 8000
```

### CUDA Out of Memory 오류

```bash
# GPU 메모리 사용률 낮추기
docker run ... \
  --gpu-memory-utilization 0.7 \  # 0.9 → 0.7
  --max-model-len 2048             # 4096 → 2048
```

### 응답이 너무 느림

```bash
# 양자화 사용
docker run ... \
  --quantization fp8
```

---

## 📚 참고 자료

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [실제 구현 코드](../3.%20code_demo/app/services/llm_service.py)
