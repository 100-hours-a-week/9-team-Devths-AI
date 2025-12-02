# AI Model Serving API

세 가지 AI 모델을 FastAPI로 서빙하는 프로젝트입니다:
1. **이미지 분류 모델**: Keras 기반 강아지/고양이 분류
2. **감성 분석 모델**: Naive Bayes 기반 텍스트 감성 분석 (positive/negative)
3. **채팅 모델**: Ollama 기반 LLM 채팅 (gemma3:4b 등)

## 프로젝트 구조

```
FASTAPI_Project_model/
├── app/
│   ├── main.py                     # FastAPI 앱 진입점
│   ├── routers/
│   │   ├── predict_routes.py       # 이미지 분류 API 라우터
│   │   ├── sentiment_routes.py     # 감성 분석 API 라우터
│   │   └── chat_routes.py          # 채팅 API 라우터
│   ├── services/
│   │   ├── model_service.py        # 이미지 분류 모델 서비스
│   │   ├── sentiment_service.py    # 감성 분석 모델 서비스
│   │   └── chat_service.py         # 채팅 모델 서비스 (Ollama)
│   ├── schemas/
│   │   └── prediction.py           # Pydantic 응답 스키마
│   └── core/
│       └── exceptions.py           # 예외 처리
├── assets/
│   ├── keras_model.h5              # Keras 이미지 분류 모델
│   └── labels.txt                  # 클래스 레이블 (Dog, Cat)
├── models/
│   └── sentiment.py                # 감성 분석 모델 구현
└── requirements.txt                # 의존성 패키지
```

## 실행 방법

### 1. 가상환경 활성화
```bash
conda activate env_python310
```

### 2. 필요한 패키지 설치
```bash
# requirements.txt 사용 (권장)
pip install -r requirements.txt

# 또는 개별 설치
pip install fastapi uvicorn tensorflow tensorflow-macos tensorflow-metal keras pillow numpy ollama python-multipart
```

**참고**: 
- Python 3.10 환경 사용 (`env_python310`)
- macOS에서 TensorFlow 사용 시: `tensorflow-macos`와 `tensorflow-metal` 설치 필요
- Ollama는 별도로 설치 및 실행되어야 합니다 (https://ollama.ai)

### 3. 서버 실행
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/FASTAPI_Project_model

# macOS에서 TensorFlow 사용 시 환경 변수 설정 (필수)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# 서버 실행 (기본 포트: 8502)
uvicorn app.main:app --host 0.0.0.0 --port 8502
```

**참고**:
- 기본 포트는 **8502**입니다 (Backend API는 8001, Streamlit은 8501 사용)
- `--reload` 옵션은 macOS에서 mutex lock 에러를 유발할 수 있으므로 프로덕션에서는 제외 권장
- 서버가 `http://localhost:8502`에서 실행됩니다

## API 테스트

### 1. 헬스 체크
```bash
curl http://localhost:8502/
```

**응답 예시:**
```json
{
  "message": "AI Model Serving API is running",
  "version": "1.0.0",
  "endpoints": {
    "image_classification": "/api/predict (POST)",
    "sentiment_analysis": "/api/sentiment (POST)",
    "chat": "/api/chat (POST)",
    "documentation": "/docs"
  }
}
```

### 2. 이미지 분류 예측 (강아지/고양이)

#### curl로 테스트
```bash
curl -X POST "http://localhost:8502/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/dog_or_cat_image.jpg"
```

**실제 예시 (프로젝트 내 이미지 사용):**
```bash
# 9week 디렉터리의 고양이 이미지로 테스트
curl -X POST "http://localhost:8502/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/yoon-dong-gyu/kakao_bootcamp/9week(20251110~20251115)/cls_cats_and_dogs/cat/cat.1.jpg"

# 강아지 이미지로 테스트
curl -X POST "http://localhost:8502/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/yoon-dong-gyu/kakao_bootcamp/9week(20251110~20251115)/cls_cats_and_dogs/dog/dog.1.jpg"
```

#### Postman으로 테스트
1. **Method**: POST
2. **URL**: `http://localhost:8502/api/predict`
3. **Body 탭**: `form-data` 선택
4. **Key**: `file` (Type을 File로 변경)
5. **Value**: 강아지 또는 고양이 이미지 파일 선택
6. **Send** 클릭

**정상 응답 예시:**
```json
{
  "class_name": "Dog",
  "confidence_score": 0.9876543
}
```

또는

```json
{
  "class_name": "Cat",
  "confidence_score": 0.9654321
}
```

### 3. 텍스트 감성 분석

#### curl로 테스트
```bash
# 기본 감성 분석 (토큰 영향도 제외)
curl -X POST "http://localhost:8502/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I really love this camera, the picture quality is amazing!",
    "explain": false
  }'

# 토큰 영향도 포함
curl -X POST "http://localhost:8502/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is the worst product ever, totally disappointed",
    "explain": true
  }'
```

#### Postman으로 테스트
1. **Method**: POST
2. **URL**: `http://localhost:8502/api/sentiment`
3. **Body 탭**: `raw` 선택, `JSON` 타입 선택
4. **Body 내용**:
```json
{
  "text": "I absolutely love this product, it works great",
  "explain": true
}
```
5. **Send** 클릭

**정상 응답 예시 (긍정):**
```json
{
  "label": "positive",
  "confidence": 0.9876543,
  "probabilities": {
    "positive": 0.9876543,
    "negative": 0.0123457
  },
  "top_tokens": [
    {"token": "love", "impact": 2.5},
    {"token": "great", "impact": 1.8},
    {"token": "absolutely", "impact": 1.2}
  ]
}
```

**정상 응답 예시 (부정):**
```json
{
  "label": "negative",
  "confidence": 0.9654321,
  "probabilities": {
    "positive": 0.0345679,
    "negative": 0.9654321
  },
  "top_tokens": [
    {"token": "worst", "impact": -3.2},
    {"token": "disappointed", "impact": -2.1}
  ]
}
```

### 4. 채팅 API (Ollama)

#### 사전 요구사항
1. **Ollama 설치 및 실행**
   ```bash
   # Ollama 설치 (https://ollama.ai)
   # 모델 다운로드
   ollama pull gemma3:4b
   ```

2. **Ollama 서버 실행 확인**
   ```bash
   # 기본 포트: 11434
   curl http://localhost:11434/api/tags
   ```

#### curl로 테스트
```bash
curl -X POST "http://localhost:8502/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "model": "gemma3:4b"
  }'
```

**응답 형식 (NDJSON 스트리밍):**
```json
{"type": "thinking_start"}
{"type": "thinking", "content": "..."}
{"type": "thinking_end"}
{"type": "content", "content": "Hello"}
{"type": "content", "content": "!"}
{"type": "content", "content": " I"}
...
```

#### Postman으로 테스트
1. **Method**: POST
2. **URL**: `http://localhost:8502/api/chat`
3. **Body 탭**: `raw` 선택, `JSON` 타입 선택
4. **Body 내용**:
```json
{
  "message": "What is machine learning?",
  "model": "gemma3:4b"
}
```
5. **Send** 클릭

**참고**: 
- 응답은 스트리밍 형식(NDJSON)으로 전송됩니다
- `model` 파라미터는 선택사항이며, 기본값은 `"gemma3:4b"`입니다
- Ollama 서버가 실행되지 않으면 연결 오류가 발생합니다

### 5. 예외 처리 테스트

#### 이미지 분류 - 파일 없이 요청
```bash
curl -X POST "http://localhost:8502/api/predict"
```

**응답 (422 Unprocessable Entity):**
```json
{
  "message": "validation_error",
  "data": {
    "details": "..."
  }
}
```

#### 이미지 분류 - 잘못된 파일 형식
```bash
curl -X POST "http://localhost:8502/api/predict" \
  -F "file=@/path/to/document.pdf"
```

**응답 (400 Bad Request):**
```json
{
  "message": "invalid_file_type",
  "data": {
    "allowed": ["jpg", "png", "jpeg"]
  }
}
```

#### 감성 분석 - 빈 텍스트
```bash
curl -X POST "http://localhost:8502/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "", "explain": false}'
```

**응답 (400 Bad Request):**
```json
{
  "message": "text_required",
  "data": null
}
```

#### 감성 분석 - 알파벳 없는 텍스트
```bash
curl -X POST "http://localhost:8502/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "123 456", "explain": false}'
```

**응답 (400 Bad Request):**
```json
{
  "message": "text must contain alphabetic characters",
  "data": null
}
```

## API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8502/docs
- ReDoc: http://localhost:8502/redoc

## 주요 기능

### 이미지 분류 모델 (Keras)
- ✅ Keras/TensorFlow 모델 로딩 (앱 시작 시)
- ✅ 이미지 파일 업로드 및 전처리
- ✅ 강아지/고양이 분류 예측
- ✅ 파일 형식 검증 (jpg, png, jpeg만 허용)

### 감성 분석 모델 (Naive Bayes)
- ✅ 메모리 기반 경량 모델 (즉시 로딩)
- ✅ 영어 텍스트 감성 분석 (positive/negative)
- ✅ 확률 분포 및 신뢰도 제공
- ✅ 토큰별 영향도 분석 (옵션)
- ✅ 입력 검증 (빈 텍스트, 알파벳 포함 여부)

### 채팅 모델 (Ollama)
- ✅ Ollama LLM 통합 (gemma3:4b 등)
- ✅ 스트리밍 응답 지원 (NDJSON 형식)
- ✅ Thinking 과정 포함 (모델 지원 시)
- ✅ 비동기 스트리밍 처리
- ✅ 다양한 모델 선택 가능

### 공통 기능
- ✅ 일관된 JSON 응답 포맷
- ✅ Pydantic 스키마 검증
- ✅ 포괄적인 예외 처리
- ✅ 자동 API 문서 생성 (Swagger UI)
- ✅ 앱 시작 시 모델 자동 로딩

## 트러블슈팅

### 모델 로딩 실패
- `assets/keras_model.h5`와 `assets/labels.txt` 파일이 존재하는지 확인
- Keras/TensorFlow가 설치되어 있는지 확인: `pip list | grep -i keras`

### 포트 충돌
- **현재 포트 구성**:
  - Backend API: 8001
  - Model API: 8502
  - Streamlit: 8501
- 다른 포트로 변경하려면: `uvicorn app.main:app --host 0.0.0.0 --port 원하는포트번호`
- Backend API가 Model API 포트를 자동 감지하므로, 포트 변경 시 Backend 재시작 필요

### 이미지 업로드 오류
- 파일 크기가 너무 크지 않은지 확인
- 파일 형식이 jpg, png, jpeg 중 하나인지 확인

### Ollama 연결 오류
- Ollama 서버가 실행 중인지 확인: `curl http://localhost:11434/api/tags`
- 필요한 모델이 다운로드되었는지 확인: `ollama list`
- 모델이 없으면 다운로드: `ollama pull gemma3:4b`
- macOS에서 mutex lock 에러 발생 시: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

