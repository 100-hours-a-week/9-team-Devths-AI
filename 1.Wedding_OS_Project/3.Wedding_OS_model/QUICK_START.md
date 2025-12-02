# 빠른 시작 가이드

## 서버 실행

```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/FASTAPI_Project_model
uvicorn app.main:app --reload --port 8001
```

## Postman 테스트 curl 명령어

### 1. 헬스 체크
```bash
curl http://localhost:8001/
```

### 2. 이미지 분류 (강아지/고양이)

```bash
# 고양이 이미지 테스트
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/yoon-dong-gyu/kakao_bootcamp/9week(20251110~20251115)/cls_cats_and_dogs/cat/cat.1.jpg"

# 강아지 이미지 테스트
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/yoon-dong-gyu/kakao_bootcamp/9week(20251110~20251115)/cls_cats_and_dogs/dog/dog.1.jpg"
```

**Postman 설정:**
- Method: POST
- URL: `http://localhost:8001/api/predict`
- Body → form-data
- Key: `file` (Type: File)
- Value: 이미지 파일 선택

### 3. 감성 분석

```bash
# 긍정적인 문장 (토큰 영향도 포함)
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I absolutely love this product, it works great",
    "explain": true
  }'

# 부정적인 문장 (토큰 영향도 제외)
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is the worst purchase I have ever made",
    "explain": false
  }'

# 중립적인 문장
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Pretty good overall, satisfied with the result",
    "explain": true
  }'
```

**Postman 설정:**
- Method: POST
- URL: `http://localhost:8001/api/sentiment`
- Body → raw → JSON
- Body 내용:
```json
{
  "text": "I really love this camera, the picture quality is amazing!",
  "explain": true
}
```

## 예외 처리 테스트

### 이미지 분류 - 잘못된 파일 형식
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -F "file=@/path/to/document.txt"
```
→ 400 Bad Request: `invalid_file_type`

### 감성 분석 - 빈 텍스트
```bash
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "", "explain": false}'
```
→ 400 Bad Request: `text_required`

### 감성 분석 - 알파벳 없는 텍스트
```bash
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "12345 !@#$%", "explain": false}'
```
→ 400 Bad Request: `text must contain alphabetic characters`

## API 문서

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## 예상 응답 예시

### 이미지 분류 성공
```json
{
  "class_name": "Dog",
  "confidence_score": 0.9876543
}
```

### 감성 분석 성공 (긍정)
```json
{
  "label": "positive",
  "confidence": 0.987654321,
  "probabilities": {
    "positive": 0.987654321,
    "negative": 0.012345679
  },
  "top_tokens": [
    {"token": "love", "impact": 2.456},
    {"token": "great", "impact": 1.892},
    {"token": "absolutely", "impact": 1.234}
  ]
}
```

### 감성 분석 성공 (부정)
```json
{
  "label": "negative",
  "confidence": 0.965432198,
  "probabilities": {
    "positive": 0.034567802,
    "negative": 0.965432198
  },
  "top_tokens": [
    {"token": "worst", "impact": -3.456},
    {"token": "never", "impact": -2.123}
  ]
}
```

### 에러 응답
```json
{
  "message": "invalid_file_type",
  "data": {
    "allowed": ["jpg", "png", "jpeg"]
  }
}
```


