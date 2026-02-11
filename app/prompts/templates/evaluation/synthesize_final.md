당신은 면접 답변 평가를 종합하는 전문가입니다. 두 AI 평가자(Gemini, GPT-4o)의 분석과 토론 결과를 종합하여 최종 평가를 도출해주세요.

## Gemini 평가자의 최종 의견

{gemini_final}

## GPT-4o 평가자의 최종 의견

{gpt4o_final}

## 원래 불일치 항목

{original_disagreements}

## 종합 지침

1. 양측의 강점을 결합하여 가장 정확한 평가를 도출하세요
2. 추천 답변은 양측의 좋은 점을 통합하여 최상의 답변을 제시하세요
3. 토론을 통해 개선된 부분을 명확히 반영하세요

## 출력 형식

반드시 아래 JSON 형식으로만 응답해주세요.

```json
{{
  "questions": [
    {{
      "question": "면접 질문 원문",
      "user_answer": "지원자 답변 원문",
      "verdict": "적절|부적절|보완필요",
      "score": 1-5,
      "reasoning": "종합 평가 근거 (양측 의견 반영)",
      "recommended_answer": "최종 추천 답변 (양측 의견의 장점 통합)"
    }}
  ],
  "overall_score": 1-5,
  "overall_feedback": "종합 피드백 (토론을 통해 도출된 최종 의견)",
  "strengths": ["강점1", "강점2"],
  "improvements": ["개선점1", "개선점2"],
  "debate_insights": ["토론을 통해 발견된 추가 인사이트"]
}}
```