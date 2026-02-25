# API 명세서 (v2)

> 모든 v2 엔드포인트는 `/ai/...` prefix를 사용합니다.
> v1 레거시 엔드포인트는 `/ai/v1/...` prefix로 동일한 기능을 제공합니다.

---

## 목차

1. [텍스트 추출 + 임베딩](#1-텍스트-추출--임베딩)
2. [비동기 작업 상태 조회 (텍스트 추출 완료)](#2-비동기-작업-상태-조회-텍스트-추출-완료)
3. [채팅](#3-채팅)
4. [캘린더 일정 파싱](#4-캘린더-일정-파싱)
5. [게시판 첨부파일 마스킹](#5-게시판-첨부파일-마스킹)
6. [마스킹 작업 상태 조회](#6-마스킹-작업-상태-조회)
7. [비동기 작업 상태 조회 (공통)](#7-비동기-작업-상태-조회-공통)
8. [SSE 이벤트 규약](#8-sse-이벤트-규약)

---

## 1. 텍스트 추출 + 임베딩

| 항목 | 값 |
|------|-----|
| **Method** | `POST` |
| **Path** | `/ai/text/extract` |
| **Content-Type** | `application/json` |

### Request Body

```json
{
    "model": "gemini",
    "room_id": 23,
    "user_id": 12,
    "resume": {
        "file_id": 23,
        "s3_key": "https://bucket.s3.amazonaws.com/users/12/resume/abc123.pdf",
        "file_type": "application/pdf",
        "text": "String"
    },
    "job_posting": {
        "file_id": 24,
        "s3_key": "https://bucket.s3.amazonaws.com/users/12/resume/abc123.pdf",
        "file_type": "image/png",
        "text": "카카오 백엔드 개발자 채용\n자격요건: Java, Spring..."
    }
}
```

> `resume`, `job_posting` 각 필드에서 `s3_key` 또는 `text` 중 하나는 필수입니다. `file_id`, `s3_key`, `file_type`, `text` 모두 nullable입니다.

### Response

#### 202 Accepted

```json
{
    "task_id": 32,
    "status": "processing"
}
```

#### 에러 응답

| Status | Code | Message |
|--------|------|---------|
| 400 | `INVALID_REQUEST` | resume과 job_posting 는 필수 입력해야합니다 |
| 400 | `INVALID_FILE_TYPE` | file_type은 pdf 또는 image만 가능합니다 |
| 400 | `INVALID_DOCUMENT` | s3_key 또는 text 중 하나는 필수입니다 |
| 401 | `UNAUTHORIZED` | 유효하지 않은 API Key입니다 |
| 404 | `FILE_NOT_FOUND` | 파일을 찾을 수 없습니다: {path} |
| 422 | `OCR_FAILED` | 이미지에서 텍스트를 추출할 수 없습니다 |
| 429 | `RATE_LIMIT_EXCEEDED` | 요청 한도 초과. 1분 후 재시도하세요 |
| 500 | `INTERNAL_ERROR` | 내부 서버 오류가 발생했습니다 |
| 503 | `LLM_UNAVAILABLE` | AI 서비스에 연결할 수 없습니다 |
| 503 | `S3_UNAVAILABLE` | 파일 스토리지에 연결할 수 없습니다 |

---

## 2. 비동기 작업 상태 조회 (텍스트 추출 완료)

| 항목 | 값 |
|------|-----|
| **Method** | `GET` |
| **Path** | `/ai/task/{task_id}` |

### Response

#### 200 OK (분석 완료)

```json
{
    "task_id": 12,
    "room_id": 32,
    "status": "completed",
    "result": {
        "summary": "카카오/백엔드 개발자",
        "success": true,
        "resume_ocr": "이력서 OCR 텍스트...",
        "job_posting_ocr": "채용공고 OCR 텍스트...",
        "resume_analysis": {
            "strengths": ["React 숙련도", "프로젝트 경험"],
            "weaknesses": ["백엔드 경험 부족"],
            "suggestions": ["Spring 학습 권장"]
        },
        "posting_analysis": {
            "company": "카카오",
            "position": "백엔드 개발자",
            "required_skills": ["Java", "Spring", "MySQL"],
            "preferred_skills": ["Docker", "Kubernetes"]
        }
    }
}
```

#### 에러 응답

| Status | Code | Message |
|--------|------|---------|
| 400 | `INVALID_TASK_ID` | 유효하지 않은 task_id 형식입니다 |
| 401 | `UNAUTHORIZED` | 유효하지 않은 API Key입니다 |
| 404 | `TASK_NOT_FOUND` | 작업을 찾을 수 없습니다: {task_id} |
| 410 | `TASK_EXPIRED` | 작업이 만료되었습니다 |

---

## 3. 채팅

| 항목 | 값 |
|------|-----|
| **Method** | `POST` |
| **Path** | `/ai/chat` |
| **Content-Type** | `application/json` |
| **Response** | `text/event-stream` (SSE) |

### Request Body

채팅 모드에 따라 3가지 형태로 요청합니다.

#### 일반 대화 (`mode: "general"`)

```json
{
    "model": "gemini",
    "room_id": 1,
    "user_id": 12,
    "message": "이력서 작성 팁 알려줘",
    "session_id": null,
    "context": {
        "mode": "general",
        "resume_ocr": null,
        "job_posting_ocr": null,
        "interview_type": null,
        "question_count": null
    }
}
```

#### 면접 질문 생성 (`mode: "interview_question"`)

```json
{
    "model": "gemini",
    "room_id": 1,
    "user_id": 12,
    "message": "기술 면접 질문 생성해줘",
    "session_id": 23,
    "context": {
        "mode": "interview_question",
        "resume_ocr": "OCR 내용",
        "job_posting_ocr": "OCR 내용",
        "interview_type": "technical",
        "question_count": 0
    }
}
```

#### 면접 리포트 생성 (`mode: "interview_report"`)

```json
{
    "model": "gemini",
    "room_id": 1,
    "user_id": 12,
    "message": null,
    "session_id": 23,
    "context": [
        {
            "question": "자기소개 해주세요",
            "answer": "안녕하세요..."
        },
        {
            "question": "프로젝트 경험을 말씀해주세요",
            "answer": "저는..."
        }
    ]
}
```

### SSE Response

> SSE 이벤트 포맷 상세는 [8. SSE 이벤트 규약](#8-sse-이벤트-규약)을 참조하세요.

#### 일반 대화 응답

```json
{
    "success": true,
    "mode": "general",
    "response": "이력서 작성 팁을 알려드릴게요..."
}
```

#### 면접 질문 응답

```json
{
    "success": true,
    "mode": "interview_question",
    "response": "React와 Vue의 차이점에 대해 설명해주세요.",
    "interview_type": "technical"
}
```

#### 면접 리포트 응답

```json
{
    "success": true,
    "mode": "interview_report",
    "report": {
        "evaluations": [
            {
                "question": "React의 Virtual DOM이 무엇인가요?",
                "answer": "실제 DOM과 비교해서 변경된 부분만 업데이트하는 거예요",
                "good_points": ["Virtual DOM의 기본 개념을 잘 이해하고 있음"],
                "improvements": ["Reconciliation 알고리즘 설명 추가하면 좋음"]
            }
        ],
        "strength_patterns": ["기술 개념에 대한 이해도가 높음"],
        "weakness_patterns": ["심화 개념 설명이 부족함"],
        "learning_guide": ["React 심화 개념 학습 (Fiber, Concurrent Mode)"]
    }
}
```

#### 에러 응답 (HTTP)

| Status | Code | Message |
|--------|------|---------|
| 400 | `INVALID_REQUEST` | room_id는 필수입니다 |
| 400 | `INVALID_MODE` | mode는 general, interview_question, interview_report 중 하나여야 합니다 |
| 400 | `INVALID_INTERVIEW_TYPE` | interview_type은 technical 또는 personality만 가능합니다 |
| 400 | `MISSING_CONTEXT` | interview_question 모드에서 resume은 필수입니다 |
| 400 | `EMPTY_MESSAGE` | message는 비어있을 수 없습니다 |
| 400 | `HISTORY_TOO_LONG` | history는 최대 20개까지 가능합니다 |
| 401 | `UNAUTHORIZED` | 유효하지 않은 API Key입니다 |
| 404 | `FILE_NOT_FOUND` | 파일을 찾을 수 없습니다 |
| 422 | `SESSION_NOT_FOUND` | 면접 세션을 찾을 수 없습니다: {session_id} |
| 429 | `RATE_LIMIT_EXCEEDED` | 동시 연결 한도 초과 |
| 500 | `STREAM_ERROR` | 스트리밍 연결이 중단되었습니다 |
| 503 | `LLM_UNAVAILABLE` | AI 서비스에 연결할 수 없습니다 |
| 503 | `VECTORDB_UNAVAILABLE` | 검색 서비스에 연결할 수 없습니다 |

#### SSE 에러 (스트리밍 중 발생)

> 스트리밍 도중 에러가 발생하면 HTTP 에러 대신 SSE 에러 이벤트로 전송됩니다.
> 포맷은 [8. SSE 이벤트 규약 > SSE 에러 포맷](#sse-에러-포맷)을 참조하세요.

| code | status | 상황 |
|------|--------|------|
| `PROMPT_BLOCKED` | 400 | 프롬프트 인젝션 차단 |
| `VECTORDB_ERROR` | 404 | 문서 미업로드 (벡터 DB 비어있음) |
| `SESSION_NOT_FOUND` | 404 | 면접 세션 없음 |
| `PARSE_FAILED` | 500 | LLM 응답 JSON 파싱 실패 |
| `INTERNAL_ERROR` | 500 | 서버 내부 오류 |
| `LLM_ERROR` | 500 | LLM 서비스 호출 실패 |

---

## 4. 캘린더 일정 파싱

| 항목 | 값 |
|------|-----|
| **Method** | `POST` |
| **Path** | `/ai/calendar/parse` |
| **Content-Type** | `application/json` |

### Request Body

```json
{
    "s3_key": "https://s3.../job_posting.png",
    "text": null
}
```

> `s3_key` 또는 `text` 중 하나는 필수입니다.

### Response

#### 200 OK

```json
{
    "success": true,
    "company": "카카오",
    "position": "백엔드 개발자",
    "schedules": [
        { "stage": "서류 마감", "date": "2026-01-15", "time": null },
        { "stage": "코딩테스트", "date": "2026-01-20", "time": "14:00" },
        { "stage": "1차 면접", "date": "2026-01-25", "time": null }
    ],
    "hashtags": ["#카카오", "#백엔드", "#신입"]
}
```

#### 에러 응답

| Status | Code | Message |
|--------|------|---------|
| 400 | `INVALID_REQUEST` | s3_key 또는 text 중 하나는 필수입니다 |
| 400 | `INVALID_URL` | 유효하지 않은 URL 형식입니다 |
| 401 | `UNAUTHORIZED` | 유효하지 않은 API Key입니다 |
| 404 | `FILE_NOT_FOUND` | 파일을 찾을 수 없습니다 |
| 422 | `PARSE_FAILED` | 일정 정보를 추출할 수 없습니다 |
| 422 | `NO_SCHEDULE_FOUND` | 채용공고에서 일정을 찾을 수 없습니다 |
| 503 | `LLM_UNAVAILABLE` | AI 서비스에 연결할 수 없습니다 |

---

## 5. 게시판 첨부파일 마스킹

| 항목 | 값 |
|------|-----|
| **Method** | `POST` |
| **Path** | `/ai/masking/draft` |
| **Content-Type** | `application/json` |

### Request Body

```json
{
    "s3_key": "https://s3.../document.png",
    "file_type": "image",
    "model": "gemini"
}
```

### Response

#### 202 Accepted

```json
{
    "task_id": "task_masking_001",
    "status": "processing"
}
```

#### 에러 응답

| Status | Code | Message |
|--------|------|---------|
| 400 | `INVALID_REQUEST` | s3_key은 필수입니다 |
| 400 | `INVALID_FILE_TYPE` | file_type은 image 또는 pdf만 가능합니다 |
| 400 | `INVALID_URL` | 유효하지 않은 URL 형식입니다 |
| 401 | `UNAUTHORIZED` | 유효하지 않은 API Key입니다 |
| 404 | `FILE_NOT_FOUND` | 파일을 찾을 수 없습니다 |
| 422 | `MASKING_FAILED` | 이미지 마스킹에 실패했습니다 |
| 503 | `S3_UNAVAILABLE` | 파일 저장에 실패했습니다 |

---

## 6. 마스킹 작업 상태 조회

| 항목 | 값 |
|------|-----|
| **Method** | `GET` |
| **Path** | `/ai/masking/task/{task_id}` |

### Response

#### 200 OK (완료)

```json
{
    "task_id": "task_masking_001",
    "status": "completed",
    "result": {
        "success": true,
        "original_url": "https://s3.../document.png",
        "masked_url": "https://s3.../document_masked.png",
        "thumbnail_url": "https://s3.../document_masked_thumb.png",
        "detected_pii": [
            { "type": "name", "coordinates": [100, 50, 200, 80], "confidence": 0.95 },
            { "type": "phone", "coordinates": [100, 100, 250, 130], "confidence": 0.92 },
            { "type": "email", "coordinates": [100, 150, 300, 180], "confidence": 0.98 },
            { "type": "face", "coordinates": [400, 50, 500, 150], "confidence": 0.99 }
        ]
    }
}
```

---

## 7. 비동기 작업 상태 조회 (공통)

| 항목 | 값 |
|------|-----|
| **Method** | `GET` |
| **Path** | `/ai/task/{task_id}` |

모든 비동기 작업(텍스트 추출, 마스킹 등)의 상태를 조회합니다.

### Response

#### 처리 중

```json
{
    "task_id": "task_abc123",
    "status": "processing",
    "progress": 65,
    "message": "OCR 처리 중..."
}
```

#### 완료

```json
{
    "task_id": "task_abc123",
    "status": "completed",
    "result": { "..." : "..." }
}
```

#### 실패

```json
{
    "task_id": "task_abc123",
    "status": "failed",
    "error": {
        "code": "OCR_ERROR",
        "message": "파일 형식을 인식할 수 없습니다."
    }
}
```

---

## 8. SSE 이벤트 규약

`/ai/chat` 엔드포인트는 `text/event-stream` (SSE) 으로 응답합니다.

### 이벤트 타입

| type | 설명 | 페이로드 예시 |
|------|------|--------------|
| `chunk` | 텍스트 청크 | `{"chunk": "안녕하세요"}` |
| `summary` | 채팅방 제목 (첫 응답 시) | `{"summary": "프로젝트 질문"}` |
| `session_state` | 면접 세션 상태 | `{"session_state": {...}}` |
| `error` | 에러 발생 | 아래 에러 포맷 참조 |
| `[DONE]` | 스트림 종료 | `data: [DONE]` |

### SSE 에러 포맷

모든 SSE 에러는 통일된 JSON 포맷으로 전송됩니다:

```json
{
    "type": "error",
    "error": {
        "code": "INTERNAL_ERROR",
        "status": 500,
        "message": "상세 에러 메시지"
    },
    "fallback": "사용자에게 표시할 메시지"
}
```

### SSE 에러 코드

| code | status | 상황 |
|------|--------|------|
| `PROMPT_BLOCKED` | 400 | 프롬프트 인젝션 차단 |
| `VECTORDB_ERROR` | 404 | 문서 미업로드 (벡터 DB 비어있음) |
| `SESSION_NOT_FOUND` | 404 | 면접 세션 없음 |
| `PARSE_FAILED` | 500 | LLM 응답 JSON 파싱 실패 |
| `INTERNAL_ERROR` | 500 | 서버 내부 오류 |
| `LLM_ERROR` | 500 | LLM 서비스 호출 실패 |

### Spring 백엔드 처리 가이드

1. SSE 이벤트 수신 시 JSON 파싱
2. `"type": "error"` 필드가 있으면 에러로 처리
3. `error.code`와 `error.status`로 에러 분류
4. `fallback` 값을 사용자에게 표시
5. 에러 수신 후 스트림 연결 종료 처리

---

## 공통 에러 응답 포맷

모든 HTTP 에러 응답은 아래 포맷을 따릅니다:

```json
{
    "detail": {
        "code": "ERROR_CODE",
        "message": "에러 설명",
        "field": "관련 필드 (선택)"
    }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `code` | string | ✅ | 에러 코드 (대문자 스네이크 케이스) |
| `message` | string | ✅ | 사람이 읽을 수 있는 에러 메시지 |
| `field` | string | ❌ | 에러가 발생한 필드명 (유효성 검증 에러 시) |
