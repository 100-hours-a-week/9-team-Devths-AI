# 구현된 AI 기능 목록

## 📊 현재 구현된 AI 기능

### 1. 감성 분석 (Sentiment Analysis)
- **위치**: `app/services/model_client.py` → `analyze_sentiment()`
- **모델**: Naive Bayes 기반 감성 분석 모델
- **API 엔드포인트**: 
  - `POST /api/model/sentiment` - 감성 분석만 수행
  - `POST /api/model/analyze` - 통합 분석 (감성 + 요약 + 태그)
- **기능**: 
  - 텍스트의 긍정/부정/중립 판단
  - 신뢰도 점수 제공
  - 토큰별 영향도 분석 (옵션)

### 2. 텍스트 요약 (Text Summarization)
- **위치**: `app/services/model_client.py` → `summarize_text()`
- **모델**: Gemini 2.5 Flash (LLM)
- **API 엔드포인트**: 
  - `POST /api/model/summarize` - 요약만 수행
  - `POST /api/model/analyze` - 통합 분석
- **기능**: 
  - 긴 텍스트를 간결하게 요약
  - 게시글 자동 요약 생성
  - 리뷰 요약 생성

### 3. 자동 태깅 (Auto Tagging)
- **위치**: `app/services/model_client.py` → `auto_tag_text()`
- **모델**: Gemini 2.5 Flash (LLM)
- **API 엔드포인트**: 
  - `POST /api/model/auto-tag` - 태그 생성만 수행
  - `POST /api/model/analyze` - 통합 분석
- **기능**: 
  - 텍스트에서 키워드 추출
  - 카테고리 자동 분류
  - 태그 자동 생성

### 4. 리뷰 요약 (Review Summary)
- **위치**: `app/services/review_summary_service.py`
- **모델**: Gemini 2.5 Flash + 감성 분석 모델
- **API 엔드포인트**: 
  - `POST /api/posts/reviews/summarize` - 게시판 리뷰 요약
  - `POST /api/vendors/{vendor_id}/reviews/summarize` - 업체별 리뷰 요약
- **기능**: 
  - 여러 리뷰를 종합하여 요약
  - 감성 분석 통계 (긍정/부정 비율)
  - 카테고리별 필터링 지원

### 5. Gemini LLM 통합
- **위치**: `app/services/model_client.py` → `chat_with_model()`
- **모델**: Google Gemini 2.5 Flash
- **API 엔드포인트**: 
  - `POST /api/gemini/chat` - HTTP 스트리밍
  - `WS /api/gemini/ws` - WebSocket 스트리밍
- **기능**: 
  - 대화형 채팅
  - 스트리밍 응답
  - 한글 답변 지원
  - 채팅 히스토리 관리

### 6. Ollama LLM 통합
- **위치**: `app/services/model_client.py` → `chat_with_model()`
- **모델**: Ollama 기반 LLM (gemma3:4b 등)
- **API 엔드포인트**: 
  - `POST /api/chat` - 채팅 API
- **기능**: 
  - 로컬 LLM 실행
  - 스트리밍 응답
  - 다양한 모델 선택

### 7. STT (Speech-to-Text)
- **위치**: `app/services/stt_service.py`
- **모델**: Whisper
- **API 엔드포인트**: 
  - `POST /api/voice/transcribe` - 음성 변환
- **기능**: 
  - 음성 파일을 텍스트로 변환
  - 음성 비서 기능 지원

### 8. OCR (Optical Character Recognition)
- **위치**: `app/services/ocr_service.py`
- **모델**: PaddleOCR / Tesseract
- **API 엔드포인트**: 
  - `POST /api/budget/ocr` - 이미지에서 텍스트 추출
- **기능**: 
  - 이미지에서 텍스트 추출
  - 예산서 이미지 처리
  - 테이블 구조화

### 9. 이미지 분류 (Image Classification)
- **위치**: `app/services/model_client.py` → `predict_image()`
- **모델**: Keras 기반 이미지 분류 모델
- **API 엔드포인트**: 
  - `POST /api/predict` - 이미지 분류
- **기능**: 
  - 강아지/고양이 분류
  - 이미지 업로드 및 전처리

### 10. RAG (Retrieval Augmented Generation)
- **위치**: `app/services/chat_service.py` → `build_rag_prompt()`
- **모델**: LLM + 컨텍스트 검색
- **API 엔드포인트**: 
  - `POST /api/chat` - RAG 기반 채팅
- **기능**: 
  - 컨텍스트 기반 답변 생성
  - 웨딩 관련 정보 검색
  - 개인화된 추천

### 11. 일정 추천 (Calendar Recommendation)
- **위치**: `app/services/calendar_service.py` → `generate_personalized_timeline()`
- **모델**: LLM 기반
- **기능**: 
  - 개인화된 일정 추천
  - 웨딩 타임라인 생성

### 12. 예산서 구조화 (Budget Structuring)
- **위치**: `app/services/budget_service.py` → `structure_text_with_llm()`
- **모델**: LLM 기반
- **기능**: 
  - OCR로 추출한 텍스트를 구조화
  - 예산서 데이터 추출
  - Excel/CSV 내보내기

## 🔗 모델 서버 구조

### 모델 서버 (3.Wedding_OS_model)
- **포트**: 8502
- **주요 라우터**:
  - `sentiment_routes.py` - 감성 분석
  - `chat_routes.py` - Ollama 채팅
  - `gemini_routes.py` - Gemini 채팅
  - `review_summary_routes.py` - 리뷰 요약

### 백엔드 서버 (2.Wedding_OS_back)
- **포트**: 8101
- **AI 관련 라우터**:
  - `ai_analysis_routes.py` - 통합 AI 분석
  - `review_summary_routes.py` - 리뷰 요약
  - `chat_routes.py` - 채팅 (RAG 포함)
  - `voice_routes.py` - 음성 처리
  - `budget_routes.py` - 예산서 처리

## 📝 사용 예시

### 감성 분석
```python
POST /api/model/sentiment
{
  "text": "이 업체 정말 좋아요!"
}
```

### 텍스트 요약
```python
POST /api/model/summarize
{
  "text": "긴 텍스트 내용..."
}
```

### 리뷰 요약
```python
POST /api/posts/reviews/summarize?board_type=couple&category=웨딩홀&limit=50
```

### 통합 분석
```python
POST /api/model/analyze
{
  "text": "리뷰 내용..."
}
# 응답: { sentiment, summary, tags }
```

## 🚀 향후 개선 사항

- [ ] Vector DB 통합 (임베딩 기반 검색)
- [ ] 멀티모달 AI (이미지 + 텍스트)
- [ ] 실시간 스트리밍 개선
- [ ] 모델 캐싱 최적화
- [ ] 배치 처리 지원

