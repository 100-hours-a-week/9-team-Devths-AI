#!/bin/bash

# 초간단 대화형 테스트 (sh 호환)
# 사용법: bash ultra_simple_interview.sh

URL="http://localhost:8000/v1/chat/completions"

echo "========================================="
echo "대화형 면접 테스트 (간단 버전)"
echo "========================================="
echo ""

#############################################
# 1. 이력서 분석
#############################################

echo "[1단계] 이력서 분석"
echo ""

RESPONSE_1=$(curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "다음 이력서와 채용공고를 분석하여 매칭 점수(0-100)와 강점, 약점을 알려주세요.\n\n이력서:\n- 이름: 김민수\n- 직무: MLOps 엔지니어\n- 경력: 4년\n- 기술: Python, Kubernetes, Docker, MLflow, Kubeflow\n- 프로젝트: ML 파이프라인 자동화, 모델 서빙 인프라 구축\n\n채용공고:\n- 회사: AI 스타트업\n- 포지션: MLOps 엔지니어\n- 경력: 3-5년\n- 필수 기술: Python, Kubernetes, Docker\n- 우대 사항: MLflow, Kubeflow"
    }],
    "max_tokens": 512,
    "temperature": 0.3
  }' | jq -r '.choices[0].message.content')

echo "AI 응답:"
echo "$RESPONSE_1"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 2. 면접 질문 (대화 히스토리 포함)
#############################################

echo "[2단계] 기술 면접 질문"
echo ""

RESPONSE_2=$(curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"다음 이력서와 채용공고를 분석하여 매칭 점수(0-100)와 강점, 약점을 알려주세요.\\n\\n이력서:\\n- 이름: 김민수\\n- 직무: MLOps 엔지니어\\n- 경력: 4년\\n- 기술: Python, Kubernetes, Docker, MLflow, Kubeflow\\n- 프로젝트: ML 파이프라인 자동화, 모델 서빙 인프라 구축\\n\\n채용공고:\\n- 회사: AI 스타트업\\n- 포지션: MLOps 엔지니어\\n- 경력: 3-5년\\n- 필수 기술: Python, Kubernetes, Docker\\n- 우대 사항: MLflow, Kubeflow\"},
      {\"role\": \"assistant\", \"content\": $(echo "$RESPONSE_1" | jq -Rs .)},
      {\"role\": \"user\", \"content\": \"좋습니다. 이제 위 이력서를 바탕으로 기술 면접 질문을 1개 해주세요.\"}
    ],
    \"max_tokens\": 512,
    \"temperature\": 0.7
  }" | jq -r '.choices[0].message.content')

echo "AI 응답 (면접 질문):"
echo "$RESPONSE_2"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 3. 사용자 답변
#############################################

echo "[3단계] 사용자 답변"
echo ""

USER_ANSWER="Kubeflow를 사용하여 ML 파이프라인을 구축했습니다. 데이터 전처리부터 모델 학습, 평가, 배포까지 전체 파이프라인을 자동화했고, Airflow를 사용하여 스케줄링과 모니터링을 구현했습니다."

echo "사용자 답변: $USER_ANSWER"
echo ""

RESPONSE_3=$(curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"다음 이력서와 채용공고를 분석하여 매칭 점수(0-100)와 강점, 약점을 알려주세요.\\n\\n이력서:\\n- 이름: 김민수\\n- 직무: MLOps 엔지니어\\n- 경력: 4년\\n- 기술: Python, Kubernetes, Docker, MLflow, Kubeflow\\n- 프로젝트: ML 파이프라인 자동화, 모델 서빙 인프라 구축\\n\\n채용공고:\\n- 회사: AI 스타트업\\n- 포지션: MLOps 엔지니어\\n- 경력: 3-5년\\n- 필수 기술: Python, Kubernetes, Docker\\n- 우대 사항: MLflow, Kubeflow\"},
      {\"role\": \"assistant\", \"content\": $(echo "$RESPONSE_1" | jq -Rs .)},
      {\"role\": \"user\", \"content\": \"좋습니다. 이제 위 이력서를 바탕으로 기술 면접 질문을 1개 해주세요.\"},
      {\"role\": \"assistant\", \"content\": $(echo "$RESPONSE_2" | jq -Rs .)},
      {\"role\": \"user\", \"content\": $(echo "$USER_ANSWER" | jq -Rs .)}
    ],
    \"max_tokens\": 512,
    \"temperature\": 0.7
  }" | jq -r '.choices[0].message.content')

echo "AI 응답:"
echo "$RESPONSE_3"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 4. 꼬리 질문
#############################################

echo "[4단계] 꼬리 질문 요청"
echo ""

RESPONSE_4=$(curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"다음 이력서와 채용공고를 분석하여 매칭 점수(0-100)와 강점, 약점을 알려주세요.\\n\\n이력서:\\n- 이름: 김민수\\n- 직무: MLOps 엔지니어\\n- 경력: 4년\\n- 기술: Python, Kubernetes, Docker, MLflow, Kubeflow\\n- 프로젝트: ML 파이프라인 자동화, 모델 서빙 인프라 구축\\n\\n채용공고:\\n- 회사: AI 스타트업\\n- 포지션: MLOps 엔지니어\\n- 경력: 3-5년\\n- 필수 기술: Python, Kubernetes, Docker\\n- 우대 사항: MLflow, Kubeflow\"},
      {\"role\": \"assistant\", \"content\": $(echo "$RESPONSE_1" | jq -Rs .)},
      {\"role\": \"user\", \"content\": \"좋습니다. 이제 위 이력서를 바탕으로 기술 면접 질문을 1개 해주세요.\"},
      {\"role\": \"assistant\", \"content\": $(echo "$RESPONSE_2" | jq -Rs .)},
      {\"role\": \"user\", \"content\": $(echo "$USER_ANSWER" | jq -Rs .)},
      {\"role\": \"assistant\", \"content\": $(echo "$RESPONSE_3" | jq -Rs .)},
      {\"role\": \"user\", \"content\": \"좋은 답변입니다. 그렇다면 위 답변을 바탕으로 꼬리 질문을 1개 해주세요.\"}
    ],
    \"max_tokens\": 512,
    \"temperature\": 0.7
  }" | jq -r '.choices[0].message.content')

echo "AI 응답 (꼬리 질문):"
echo "$RESPONSE_4"
echo ""
echo "========================================="
echo ""

echo "테스트 완료!"
echo ""
echo "=== 평가 포인트 ==="
echo "1. 맥락 유지: AI가 '위 이력서', '위 답변'을 이해했는가?"
echo "2. 꼬리 질문: 이전 답변과 연관성 있는 질문인가?"
echo "3. 한국어 품질: 자연스러운 한국어인가?"
echo ""
