# ADR-041: vLLM 모델 선정 (미완료)

> **작성 상태**: 🚧 초안 작성 중  
> **최종 업데이트**: 2026-01-19  
> **상태**: Proposed

---

## Context (배경)

vLLM을 사용하기로 결정했으므로, 구체적인 모델을 선정해야 합니다.

### 요구사항
- **한국어 성능**: 이력서/채용공고 분석 정확도 90% 이상
- **모델 크기**: 8B 이하 (T4 GPU 16GB 메모리 제약)
- **라이선스**: 상업적 사용 가능
- **커뮤니티 지원**: 활발한 업데이트 및 문서화

### 후보 모델
1. **Llama-3-Korean-Bllossom-8B** (MLP-KTLim)
2. **EXAONE-3.0-7.8B-Instruct** (LG AI Research)
3. **Qwen2.5-7B-Instruct** (Alibaba)

---

## Decision (결정)

### V3: Llama-3-Korean-Bllossom-8B
- **근거**: 한국어 특화, 활발한 커뮤니티
- **라이선스**: Llama 3 Community License (상업적 사용 검토 필요)

### V4+: EXAONE-3.0-7.8B-Instruct (검토 중)
- **근거**: Apache 2.0 라이선스, LG AI Research 개발
- **장점**: 상업적 사용 명확, 한국어 성능 우수

---

## Consequences (결과)

### 긍정적 영향
- ✅ 한국어 특화 모델로 정확도 향상
- ✅ 모델 교체 용이성 (vLLM 호환)
- ✅ 커뮤니티 지원 활발

### 부정적 영향
- ❌ 8B 모델 성능 제한 (Gemini 대비)
- ❌ 라이선스 검토 필요 (Llama 3)
- ❌ 모델 파인튜닝 어려움 (리소스 부족)

### 완화 전략
- EXAONE 벤치마크 테스트 (V4 전환 대비)
- 라이선스 법무 검토
- Gemini Fallback 유지

---

## Alternatives Considered (고려한 대안)

### 1. EXAONE-3.0-7.8B-Instruct
- **장점**: Apache 2.0, 한국어 우수
- **단점**: 커뮤니티 작음
- **결론**: V4에서 재검토

### 2. Qwen2.5-7B-Instruct
- **장점**: 벤치마크 성능 우수
- **단점**: 한국어 특화 아님
- **결론**: 보류

---

## Testing Plan (테스트 계획)

### 벤치마크 테스트
- [ ] 한국어 QA 데이터셋 (KorQuAD)
- [ ] 이력서 분석 정확도 (자체 데이터)
- [ ] 응답 속도 (평균, P95)

### 비교 대상
- Llama-3-Korean-8B vs EXAONE-7.8B vs Gemini 2.0 Flash

---

## Related ADRs
- ADR-040: LLM 선정
- ADR-045: RAG 전략

---

## References
- [Llama-3-Korean Model Card](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)
- [EXAONE Model Card](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [Model Select/01_LLM_선정.md](../Model%20Select/01_LLM_선정(미완료).md)
