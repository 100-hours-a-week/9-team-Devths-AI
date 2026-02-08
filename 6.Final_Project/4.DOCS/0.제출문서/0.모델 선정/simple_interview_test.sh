#!/bin/bash

# 대화형 꼬리 질문 테스트 (수정 버전)
# 사용법: ./simple_interview_test.sh

VLLM_URL="http://localhost:8000/v1/chat/completions"

echo "========================================="
echo "대화형 면접 테스트"
echo "========================================="
echo ""

# 대화 히스토리 파일
HISTORY_FILE="conversation_history.json"
echo '[]' > "$HISTORY_FILE"

# 함수: AI에게 질문하고 응답 받기
ask_ai() {
    local user_message="$1"
    
    # 현재 히스토리 읽기
    history=$(cat "$HISTORY_FILE")
    
    # 사용자 메시지 추가 (jq -Rs로 이스케이프 처리)
    history=$(echo "$history" | jq --arg msg "$user_message" '. += [{"role": "user", "content": $msg}]')
    
    # API 호출
    response=$(curl -s -X POST "$VLLM_URL" \
        -H "Content-Type: application/json" \
        -d "$(echo "$history" | jq -c '{messages: ., max_tokens: 512, temperature: 0.7}')")
    
    # AI 응답 추출
    ai_message=$(echo "$response" | jq -r '.choices[0].message.content // "ERROR: No response"')
    
    # AI 응답을 히스토리에 추가
    history=$(echo "$history" | jq --arg msg "$ai_message" '. += [{"role": "assistant", "content": $msg}]')
    
    # 히스토리 저장
    echo "$history" > "$HISTORY_FILE"
    
    # 응답 출력
    echo "$ai_message"
}

#############################################
# 1. 이력서 분석
#############################################

echo "[1단계] 이력서 분석"
echo ""

read -r -d '' PROMPT_1 << 'EOF'
다음 이력서와 채용공고를 분석하여 매칭 점수(0-100)와 강점, 약점을 알려주세요.

이력서:
- 이름: 김민수
- 직무: MLOps 엔지니어
- 경력: 4년
- 기술: Python, Kubernetes, Docker, MLflow, Kubeflow
- 프로젝트: ML 파이프라인 자동화, 모델 서빙 인프라 구축

채용공고:
- 회사: AI 스타트업
- 포지션: MLOps 엔지니어
- 경력: 3-5년
- 필수 기술: Python, Kubernetes, Docker
- 우대 사항: MLflow, Kubeflow
EOF

echo "질문: [이력서 분석 요청]"
echo ""
echo "AI 응답:"
ask_ai "$PROMPT_1"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 2. 기술 면접 질문 요청
#############################################

echo "[2단계] 기술 면접 질문"
echo ""

PROMPT_2="좋습니다. 이제 위 이력서를 바탕으로 기술 면접 질문을 1개 해주세요."

echo "질문: $PROMPT_2"
echo ""
echo "AI 응답:"
ask_ai "$PROMPT_2"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 3. 사용자 답변 (시뮬레이션)
#############################################

echo "[3단계] 사용자 답변"
echo ""

read -r -d '' USER_ANSWER << 'EOF'
Kubeflow를 사용하여 ML 파이프라인을 구축했습니다. 데이터 전처리부터 모델 학습, 평가, 배포까지 전체 파이프라인을 자동화했고, Airflow를 사용하여 스케줄링과 모니터링을 구현했습니다.
EOF

echo "사용자 답변: $USER_ANSWER"
echo ""
echo "AI 응답:"
ask_ai "$USER_ANSWER"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 4. 꼬리 질문 요청
#############################################

echo "[4단계] 꼬리 질문"
echo ""

PROMPT_4="좋은 답변입니다. 그렇다면 위 답변을 바탕으로 꼬리 질문을 1개 해주세요."

echo "질문: $PROMPT_4"
echo ""
echo "AI 응답 (꼬리 질문):"
ask_ai "$PROMPT_4"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 5. 사용자 답변 2
#############################################

echo "[5단계] 사용자 답변 2"
echo ""

read -r -d '' USER_ANSWER_2 << 'EOF'
가장 어려웠던 점은 GPU 리소스 관리였습니다. 여러 실험이 동시에 실행될 때 GPU 할당과 스케줄링이 복잡했고, Kubernetes의 GPU Sharing 기능을 활용하여 해결했습니다.
EOF

echo "사용자 답변: $USER_ANSWER_2"
echo ""
echo "AI 응답:"
ask_ai "$USER_ANSWER_2"
echo ""
echo "========================================="
echo ""
sleep 2

#############################################
# 6. 새로운 질문
#############################################

echo "[6단계] 새로운 주제 질문"
echo ""

PROMPT_6="좋습니다. 이제 다른 주제로 넘어가서, 모델 서빙 인프라에 대해 질문해주세요."

echo "질문: $PROMPT_6"
echo ""
echo "AI 응답 (새로운 질문):"
ask_ai "$PROMPT_6"
echo ""
echo "========================================="
echo ""

#############################################
# 완료
#############################################

echo ""
echo "테스트 완료!"
echo ""
echo "대화 히스토리 확인:"
echo "  cat $HISTORY_FILE | jq '.'"
echo ""
echo "대화 흐름 요약:"
cat "$HISTORY_FILE" | jq -r '.[] | "[\(.role)] \(.content | .[0:80])..."' 2>/dev/null || echo "히스토리 파일 확인 필요"
echo ""
