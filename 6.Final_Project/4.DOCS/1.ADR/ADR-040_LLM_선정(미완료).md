# ADR-040: LLM 선정 (Gemini vs vLLM) (미완료)

> **작성 상태**: 🚧 초안 작성 중  
> **최종 업데이트**: 2026-01-19  
> **상태**: Accepted

---

## Context (배경)

프로젝트에서 사용할 LLM(Large Language Model)을 선정해야 합니다.

### 주요 고려사항
- **비용**: API 호출 비용 vs GPU 서버 비용
- **한국어 성능**: 이력서/채용공고 분석 정확도
- **응답 속도**: 실시간 대화 요구사항
- **확장성**: 사용자 증가 시 대응

### 후보 모델
1. **Gemini 2.0 Flash** (Google API)
2. **vLLM** (오픈소스, Llama-3-Korean-8B)

---

## Decision (결정)

### V1-V2: Gemini 2.0 Flash
- **근거**: 빠른 프로토타이핑, 초기 비용 최소화
- **비용**: 월 $15 (사용자 10명 기준)

### V3+: vLLM (Llama-3-Korean-Bllossom-8B)
- **근거**: 비용 절감 (월 1,000+ 요청 시)
- **비용**: 월 $266 (GPU 서버 고정 비용)

### 손익분기점
- **Gemini**: $0.075/1M tokens (입력)
- **vLLM**: $266/월 (고정)
- **손익분기점**: 약 3.5B tokens/월

**결론**: 월 1B tokens 이상 사용 시 vLLM으로 전환

---

## Consequences (결과)

### 긍정적 영향
- ✅ V1-V2에서 빠른 개발 가능 (Gemini API)
- ✅ V3에서 비용 절감 (vLLM)
- ✅ 모델 교체 용이성 (동일한 OpenAI API 스펙)

### 부정적 영향
- ❌ V3 전환 시 인프라 구축 필요 (GPU 서버)
- ❌ vLLM 유지보수 부담 (모델 업데이트, 모니터링)
- ❌ 8B 모델 성능 제한 (Gemini 대비)

### 완화 전략
- Fallback 구현 (vLLM → Gemini)
- 모니터링 대시보드 구축
- 모델 성능 벤치마크 정기 실시

---

## Alternatives Considered (고려한 대안)

### 1. OpenAI GPT-4o-mini
- **장점**: 높은 성능
- **단점**: Gemini보다 2배 비싼 비용
- **결론**: 비용 대비 성능 낮음

### 2. Claude 3 Haiku
- **장점**: 빠른 응답 속도
- **단점**: 한국어 성능 제한적
- **결론**: 한국어 특화 필요

---

## Related ADRs
- ADR-041: vLLM 모델 선정
- ADR-042: OCR 전략
- ADR-045: RAG 전략

---

## References
- [Gemini API Pricing](https://ai.google.dev/pricing)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Model Select/01_LLM_선정.md](../Model%20Select/01_LLM_선정(미완료).md)
