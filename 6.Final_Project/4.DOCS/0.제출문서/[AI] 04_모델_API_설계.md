# [AI] 04_모델_API_설계

# AI Server 내부 API 명세서

> **프로젝트:** Devths AI 취업 도우미
> **작성일:** 2026-01-13
> **API 버전:** v1.9
> **마지막 업데이트:** 2026-02-08

---

## 개요

이 문서는 **Backend(Spring Boot) → AI Server(FastAPI)** 간의 내부 통신 API를 정의합니다.

```
[Frontend] → [Backend] → [AI Server] → [VectorDB/LLM]
                ↑
          이 구간의 API
```

---

## Table of Contents

### API 엔드포인트 목록

| # | 기능 | Method | Endpoint | 처리방식 |
|---|------|--------|----------|----------|
| 1 | [텍스트 추출 + 임베딩](#1-텍스트-추출--임베딩) | POST | `/ai/text/extract` | 비동기 |
| 2 | [채팅 (대화/면접 질문/면접 리포트)](#2-채팅-대화면접-질문면접-리포트) | POST | `/ai/chat` | 스트리밍 (SSE) |
| 3 | [캘린더 일정 파싱](#3-캘린더-일정-파싱) | POST | `/ai/calendar/parse` | 동기 |
| 4 | [게시판 첨부파일 마스킹](#4-게시판-첨부파일-마스킹) | POST | `/ai/masking/draft` | 비동기 |
| 5 | [비동기 작업 상태 조회](#5-비동기-작업-상태-조회) | GET | `/ai/task/{task_id}` | 동기 |
| 6 | [면접 답변 분석 (1단계)](#6-면접-답변-분석-1단계) | POST | `/ai/evaluation/analyze` | 동기 |
| 7 | [심층 분석 - 토론 (2단계)](#7-심층-분석---토론-2단계) | POST | `/ai/evaluation/debate` | 동기 |

---

## 공통 사항

### 인증

```http
X-API-Key: your-api-key-here
```

### Base URL

```
http://ai-server:8000
```

---

## 1. 텍스트 추출 + 임베딩

### 개요

이력서와 채용공고를 함께 받아 OCR 처리 후 임베딩 저장 및 분석 리포트를 생성합니다.

> **처리 흐름:** resume OCR → job_posting OCR → VectorDB 저장 → 분석 리포트 생성 → 응답
> **응답 활용:** Backend는 OCR 텍스트를 RDB에 저장하고, 분석 리포트를 채팅 메시지로 표시

### Endpoint

```
POST /ai/text/extract
```

### 처리 방식

**비동기** - task_id 반환 → 폴링 필요

### Request Headers

| Header | 값 | 필수 |
|--------|-----|:----:|
| X-API-Key | your-api-key-here | ✅ |
| Content-Type | application/json | ✅ |

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
        "text": null
    },
    "job_posting": {
        "file_id": null,
        "s3_key": null,
        "file_type": null,
        "text": "카카오 백엔드 개발자 채용\n자격요건: Java, Spring..."
    }
}
```

**사용 예시:**
- **파일 업로드**: `s3_key` + `file_type` 제공, `text`는 `null`
- **텍스트 입력**: `text` 제공, `s3_key`와 `file_type`은 `null`
- **혼합**: `resume`은 파일, `job_posting`은 텍스트 (또는 그 반대)

### Request Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| model | string | ❌ | `gemini`(기본값), `openai`, `vllm` |
| room_id | int | ✅ | 채팅방 ID |
| user_id | int | ✅ | 사용자 ID |
| resume | object | ✅ | 이력서/포트폴리오 정보 |
| job_posting | object | ✅ | 채용공고 정보 |

> ⚠️ `resume`과 `job_posting`은 필수 입력해야 합니다

### Document Object (resume / job_posting)

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| file_id | int \| null | ❌ | 파일 ID (파일 업로드 시 사용, 선택사항) |
| s3_key | string \| null | ❌ | S3 파일 URL 또는 키 (파일 업로드 시 사용) |
| file_type | string \| null | ❌ | `pdf`, `image` (파일 업로드 시 사용) |
| text | string \| null | ❌ | 직접 입력 텍스트 (텍스트 입력 시 사용) |

**입력 규칙:**
- ⚠️ 각 문서에서 `s3_key` + `file_type` 또는 `text` 중 하나는 필수입니다
- ⚠️ `s3_key`와 `text`를 동시에 사용할 수 없습니다
- ⚠️ `s3_key` 사용 시 `file_type`은 필수입니다

### Response (202 Accepted)

```json
{
    "task_id": 32,
    "status": "processing"
}
```

### Error Responses

#### 400 Bad Request

**resume과 job_posting 필수 입력 오류:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "resume과 job_posting 는 필수 입력해야합니다"
    }
}
```

**잘못된 file_type:**
```json
{
    "detail": {
        "code": "INVALID_FILE_TYPE",
        "message": "file_type은 pdf 또는 image만 가능합니다",
        "field": "resume.file_type"
    }
}
```

**s3_key 또는 text 필수 오류:**
```json
{
    "detail": {
        "code": "INVALID_DOCUMENT",
        "message": "s3_key 또는 text 중 하나는 필수입니다",
        "field": "resume"
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 404 Not Found

```json
{
    "detail": {
        "code": "FILE_NOT_FOUND",
        "message": "파일을 찾을 수 없습니다: users/12/resume/abc123.pdf"
    }
}
```

#### 422 Unprocessable Entity

```json
{
    "detail": {
        "code": "OCR_FAILED",
        "message": "이미지에서 텍스트를 추출할 수 없습니다"
    }
}
```

#### 429 Too Many Requests

```json
{
    "detail": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "요청 한도 초과. 1분 후 재시도하세요"
    }
}
```

#### 500 Internal Server Error

```json
{
    "detail": {
        "code": "INTERNAL_ERROR",
        "message": "내부 서버 오류가 발생했습니다"
    }
}
```

#### 503 Service Unavailable

**LLM 서비스 불가:**
```json
{
    "detail": {
        "code": "LLM_UNAVAILABLE",
        "message": "AI 서비스에 연결할 수 없습니다"
    }
}
```

**S3 스토리지 불가:**
```json
{
    "detail": {
        "code": "S3_UNAVAILABLE",
        "message": "파일 스토리지에 연결할 수 없습니다"
    }
}
```

### Polling 완료 시 (GET /ai/task/{task_id})

**분석 완료 응답:**

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

### Response Fields

| 필드 | 타입 | 설명 |
|------|------|------|
| task_id | int | 작업 ID |
| room_id | int | 채팅방 ID |
| status | string | `processing`, `completed`, `failed` |
| result.success | boolean | 성공 여부 |
| result.resume_ocr | string | 이력서 OCR 텍스트 |
| result.job_posting_ocr | string | 채용공고 OCR 텍스트 |
| result.resume_analysis | object | 이력서 분석 결과 |
| result.posting_analysis | object | 채용공고 분석 결과 |

---

## 2. 채팅 (대화/면접 질문/면접 리포트)

### 개요

모든 LLM 응답을 통합 처리합니다. `context.mode`로 기능을 구분합니다.

### Endpoint

```
POST /ai/chat
```

### 처리 방식

**스트리밍 (SSE)** - Server-Sent Events로 실시간 응답

### Request Headers

| Header | 값 | 필수 |
|--------|-----|:----:|
| X-API-Key | your-api-key-here | ✅ |
| Content-Type | application/json | ✅ |

### Request Body - 일반 대화 (mode: general)

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

### Request Body - 면접 질문 생성 (mode: interview_question)

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

### Request Body - 면접 리포트 생성 (면접 종료)

```json
{
    "model": "gemini",
    "room_id": 1,
    "user_id": 12,
    "session_id": 23,
    "context": [
        {
            "question": "~~",
            "answer": "...."
        }
    ]
}
```

### Request Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| model | string \| null | ❌ | `gemini`(기본값), `vllm` |
| room_id | int | ✅ | 채팅방 ID |
| user_id | int | ✅ | 사용자 ID |
| message | string \| null | ❌ | 사용자 메시지 (일반 대화 시 필수) |
| session_id | int \| null | ❌ | 면접 세션 ID (면접 모드 시 필수) |
| context | object \| array | ❌ | 채팅 컨텍스트 또는 Q&A 배열 |

### Context Object (일반 대화/면접 질문)

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| mode | string | ❌ | `general`(기본값), `interview_question`, `interview_report` |
| resume_ocr | string \| null | ❌ | 이력서 OCR 텍스트 (면접 질문 생성 시) |
| job_posting_ocr | string \| null | ❌ | 채용공고 OCR 텍스트 (면접 질문 생성 시) |
| interview_type | string \| null | ❌ | `technical`, `personality` (면접 모드 시) |
| question_count | int \| null | ❌ | 현재까지 생성된 질문 수 |

### Context Array (면접 리포트)

면접 리포트 생성 시 `context`는 Q&A 배열입니다:

```json
[
    { "question": "질문1", "answer": "답변1" },
    { "question": "질문2", "answer": "답변2" }
]
```

### ChatMode 설명

| Mode | 설명 | 사용 시점 |
|------|------|----------|
| `general` | 일반 대화 (기본값) | 취업 관련 질문, RAG 검색 |
| `interview_question` | 면접 질문 생성 | 면접 모드 시작 시, 꼬리질문 생성 시 |
| `interview_report` | 면접 리포트 생성 | 면접 종료 시 평가 및 피드백 |

### SSE Response 형식

**일반 대화:**

```json
{
    "success": true,
    "mode": "general",
    "response": "이력서 작성 팁을 알려드릴게요..."
}
```

**면접 질문:**

```json
{
    "success": true,
    "mode": "interview_question",
    "response": "React와 Vue의 차이점에 대해 설명해주세요.",
    "interview_type": "technical"
}
```

**면접 리포트:**

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

### Error Responses

#### 400 Bad Request

**필수 필드 누락:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "room_id는 필수입니다",
        "field": "room_id"
    }
}
```

**잘못된 mode:**
```json
{
    "detail": {
        "code": "INVALID_MODE",
        "message": "mode는 general, interview_question, interview_report 중 하나여야 합니다",
        "field": "context.mode"
    }
}
```

**잘못된 면접 타입:**
```json
{
    "detail": {
        "code": "INVALID_INTERVIEW_TYPE",
        "message": "interview_type은 technical 또는 personality만 가능합니다",
        "field": "context.interview_type"
    }
}
```

**필수 context 누락:**
```json
{
    "detail": {
        "code": "MISSING_CONTEXT",
        "message": "interview_question 모드에서 resume은 필수입니다",
        "field": "context.resume"
    }
}
```

**빈 메시지:**
```json
{
    "detail": {
        "code": "EMPTY_MESSAGE",
        "message": "message는 비어있을 수 없습니다",
        "field": "message"
    }
}
```

**history 초과:**
```json
{
    "detail": {
        "code": "HISTORY_TOO_LONG",
        "message": "history는 최대 20개까지 가능합니다",
        "field": "history"
    }
}
```

**프롬프트 인젝션 차단:**
```json
{
    "detail": {
        "code": "PROMPT_BLOCKED",
        "message": "프롬프트 인젝션을 차단합니다. 올바른 질문을 해주세요."
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 404 Not Found

**파일 없음:**
```json
{
    "detail": {
        "code": "FILE_NOT_FOUND",
        "message": "파일을 찾을 수 없습니다"
    }
}
```

**문서 미업로드 (VectorDB):**
```json
{
    "detail": {
        "code": "VECTORDB_ERROR",
        "message": "문서를 미업로드하였습니다."
    }
}
```

**면접 세션 없음:**
```json
{
    "detail": {
        "code": "SESSION_NOT_FOUND",
        "message": "면접 세션이 없습니다."
    }
}
```

#### 422 Unprocessable Entity

```json
{
    "detail": {
        "code": "SESSION_NOT_FOUND",
        "message": "면접 세션을 찾을 수 없습니다: interview_001"
    }
}
```

#### 429 Too Many Requests

```json
{
    "detail": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "동시 연결 한도 초과"
    }
}
```

#### 500 Internal Server Error

**스트리밍 오류:**
```json
{
    "detail": {
        "code": "STREAM_ERROR",
        "message": "스트리밍 연결이 중단되었습니다"
    }
}
```

**LLM 응답 파싱 실패:**
```json
{
    "detail": {
        "code": "PARSE_FAILED",
        "message": "LLM 응답 JSON 파싱 실패입니다."
    }
}
```

**서버 내부 오류:**
```json
{
    "detail": {
        "code": "INTERNAL_ERROR",
        "message": "서버 내부 오류입니다."
    }
}
```

**LLM 서비스 호출 실패:**
```json
{
    "detail": {
        "code": "LLM_ERROR",
        "message": "LLM 서비스 호출 실패하였습니다."
    }
}
```

#### 503 Service Unavailable

**LLM 서비스 불가:**
```json
{
    "detail": {
        "code": "LLM_UNAVAILABLE",
        "message": "AI 서비스에 연결할 수 없습니다"
    }
}
```

**VectorDB 서비스 불가:**
```json
{
    "detail": {
        "code": "VECTORDB_UNAVAILABLE",
        "message": "검색 서비스에 연결할 수 없습니다"
    }
}
```

---

## 3. 캘린더 일정 파싱

### 개요

채용공고 파일/텍스트를 분석하여 일정 정보를 추출합니다.

### Endpoint

```
POST /ai/calendar/parse
```

### 처리 방식

**동기** - 즉시 응답 반환 (Gemini Flash API 사용)

### Request Headers

| Header | 값 | 필수 |
|--------|-----|:----:|
| X-API-Key | your-api-key-here | ✅ |

### Request Body

```json
{
    "s3_key": "https://s3.../job_posting.png",
    "text": null
}
```

또는

```json
{
    "s3_key": null,
    "text": "카카오 백엔드 개발자 채용\n서류마감: 2026-01-15\n코딩테스트: 2026-01-20..."
}
```

### Request Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| s3_key | string \| null | ⚠️ | 채용공고 파일 S3 URL |
| text | string \| null | ⚠️ | 채용공고 텍스트 |

> ⚠️ `s3_key` 또는 `text` 중 하나는 필수

### Response (200 OK)

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

### Response Fields

| 필드 | 타입 | 설명 |
|------|------|------|
| success | boolean | 성공 여부 |
| company | string | 회사명 |
| position | string | 포지션 |
| schedules | array | 전형 일정 목록 |
| schedules[].stage | string | 전형 단계 |
| schedules[].date | string | 날짜 (YYYY-MM-DD) |
| schedules[].time | string \| null | 시간 (HH:MM) |
| hashtags | array | 추출된 해시태그 |

### Error Responses

#### 400 Bad Request

**필수 필드 누락:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "s3_key 또는 text 중 하나는 필수입니다"
    }
}
```

**잘못된 URL:**
```json
{
    "detail": {
        "code": "INVALID_URL",
        "message": "유효하지 않은 URL 형식입니다",
        "field": "s3_key"
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 404 Not Found

```json
{
    "detail": {
        "code": "FILE_NOT_FOUND",
        "message": "파일을 찾을 수 없습니다"
    }
}
```

#### 422 Unprocessable Entity

**파싱 실패:**
```json
{
    "detail": {
        "code": "PARSE_FAILED",
        "message": "일정 정보를 추출할 수 없습니다"
    }
}
```

**일정 없음:**
```json
{
    "detail": {
        "code": "NO_SCHEDULE_FOUND",
        "message": "채용공고에서 일정을 찾을 수 없습니다"
    }
}
```

#### 503 Service Unavailable

```json
{
    "detail": {
        "code": "LLM_UNAVAILABLE",
        "message": "AI 서비스에 연결할 수 없습니다"
    }
}
```

---

## 4. 게시판 첨부파일 마스킹

### 개요

게시판 첨부파일에서 개인정보(이름, 전화번호, 이메일, 얼굴)를 감지하고 마스킹합니다.

### Endpoint

```
POST /ai/masking/draft
```

### 처리 방식

**비동기** - task_id 반환 → 폴링 필요

### Request Headers

| Header | 값 | 필수 |
|--------|-----|:----:|
| X-API-Key | your-api-key-here | ✅ |

### Request Body

```json
{
    "s3_key": "https://s3.../document.png",
    "file_type": "image",
    "model": "gemini"
}
```

### Request Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| s3_key | string | ✅ | 원본 파일 S3 URL |
| file_type | string | ✅ | `image`, `pdf` |
| model | string | ❌ | `gemini`(기본값), `openai`, `vllm` |

### Response (202 Accepted)

```json
{
    "task_id": 32,
    "status": "processing"
}
```

### Error Responses

#### 400 Bad Request

**필수 필드 누락:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "s3_key은 필수입니다",
        "field": "s3_key"
    }
}
```

**잘못된 파일 타입:**
```json
{
    "detail": {
        "code": "INVALID_FILE_TYPE",
        "message": "file_type은 image 또는 pdf만 가능합니다",
        "field": "file_type"
    }
}
```

**잘못된 URL:**
```json
{
    "detail": {
        "code": "INVALID_URL",
        "message": "유효하지 않은 URL 형식입니다",
        "field": "s3_key"
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 404 Not Found

```json
{
    "detail": {
        "code": "FILE_NOT_FOUND",
        "message": "파일을 찾을 수 없습니다"
    }
}
```

#### 422 Unprocessable Entity

```json
{
    "detail": {
        "code": "MASKING_FAILED",
        "message": "이미지 마스킹에 실패했습니다"
    }
}
```

#### 503 Service Unavailable

```json
{
    "detail": {
        "code": "S3_UNAVAILABLE",
        "message": "파일 저장에 실패했습니다"
    }
}
```

### Polling 완료 시 (GET /ai/task/{task_id})

```json
{
    "task_id": 32,
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

### PII Type

| Type | 설명 |
|------|------|
| name | 이름 |
| phone | 전화번호 |
| email | 이메일 |
| face | 얼굴 |

---

## 5. 비동기 작업 상태 조회

### 개요

비동기 처리 작업의 상태를 조회하고 결과를 확인합니다.

### Endpoint

```
GET /ai/task/{task_id}
```

### Path Parameters

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| task_id | string | ✅ | 작업 ID |

### Response - 처리 중

```json
{
    "task_id": 32,
    "status": "processing",
    "progress": 65,
    "message": "OCR 처리 중..."
}
```

### Response - 완료

```json
{
    "task_id": 32,
    "status": "completed",
    "result": { ... }
}
```

> **참고:** `result` 내용은 작업 유형에 따라 다릅니다:
> - 텍스트 추출 + 분석: `resume_ocr`, `job_posting_ocr`, `resume_analysis`, `posting_analysis` 포함
> - 마스킹: `original_url`, `masked_url`, `thumbnail_url`, `detected_pii` 포함

### Response - 실패

```json
{
    "task_id": 32,
    "status": "failed",
    "error": {
        "code": "OCR_ERROR",
        "message": "파일 형식을 인식할 수 없습니다."
    }
}
```

### Status 값

| Status | 설명 |
|--------|------|
| processing | 처리 중 |
| completed | 완료 |
| failed | 실패 |

### Error Responses

#### 400 Bad Request

```json
{
    "detail": {
        "code": "INVALID_TASK_ID",
        "message": "유효하지 않은 task_id 형식입니다"
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 404 Not Found

```json
{
    "detail": {
        "code": "TASK_NOT_FOUND",
        "message": "작업을 찾을 수 없습니다: task_abc123"
    }
}
```

#### 410 Gone

```json
{
    "detail": {
        "code": "TASK_EXPIRED",
        "message": "작업이 만료되었습니다"
    }
}
```

### 폴링 권장 사항

- **초기 폴링 간격**: 1초
- **최대 대기 시간**: 300초 (5분)
- **지수 백오프**: 실패 시 간격을 점진적으로 증가 (1초 → 2초 → 4초 → 8초)

---

## 6. 면접 답변 분석 (1단계)

### 개요

면접 종료 시 Gemini 3 Pro (thinking 모드)가 전체 Q&A를 분석하여 각 답변의 적절성을 평가하고 피드백을 생성합니다.

> **처리 흐름:** 면접 종료 → 전체 Q&A + 이력서/채용공고 전달 → Gemini 3 Pro 분석 → 개별 평가 + 종합 피드백 반환
> **사용 모델:** `gemini-3-pro-preview` (thinking_level: HIGH)

### Endpoint

```
POST /ai/evaluation/analyze
```

### 처리 방식

**동기** - 즉시 응답 반환 (Gemini 3 Pro thinking 사용으로 10~30초 소요 가능)

### Request Headers

| Header | 값 | 필수 |
|--------|-----|:----:|
| X-API-Key | your-api-key-here | ✅ |
| Content-Type | application/json | ✅ |

### Request Body

```json
{
    "session_id": "session_abc123",
    "qa_pairs": [
        {
            "question": "Spring Boot에서 의존성 주입이 무엇인가요?",
            "answer": "의존성 주입은 객체가 필요로 하는 의존 객체를 외부에서 주입하는 패턴입니다...",
            "category": "cs_fundamentals"
        },
        {
            "question": "REST API 설계 시 주의할 점은?",
            "answer": "HTTP 메서드를 적절히 사용하고 URI는 리소스 기반으로 설계해야 합니다...",
            "category": "backend"
        }
    ],
    "resume_text": "3년 경력 백엔드 개발자...",
    "job_posting_text": "Spring Boot 경험 필수...",
    "interview_type": "tech"
}
```

### Request Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| session_id | string | ✅ | 면접 세션 ID |
| qa_pairs | array | ✅ | 질의응답 목록 |
| qa_pairs[].question | string | ✅ | 면접 질문 |
| qa_pairs[].answer | string | ✅ | 지원자 답변 |
| qa_pairs[].category | string | ❌ | 질문 카테고리 (기본값: "") |
| resume_text | string | ❌ | 이력서 텍스트 (기본값: "") |
| job_posting_text | string | ❌ | 채용공고 텍스트 (기본값: "") |
| interview_type | string | ❌ | 면접 유형: `tech`, `behavior` (기본값: "tech") |

### Response (200 OK)

```json
{
    "success": true,
    "session_id": "session_abc123",
    "questions": [
        {
            "question": "Spring Boot에서 의존성 주입이 무엇인가요?",
            "user_answer": "의존성 주입은 객체가 필요로 하는 의존 객체를 외부에서 주입하는 패턴입니다...",
            "verdict": "적절",
            "score": 4,
            "reasoning": "DI의 핵심 개념을 정확하게 설명했으나, @Autowired 등 구체적 예시가 부족합니다.",
            "recommended_answer": null,
            "category": "cs_fundamentals"
        },
        {
            "question": "REST API 설계 시 주의할 점은?",
            "user_answer": "HTTP 메서드를 적절히 사용하고 URI는 리소스 기반으로 설계해야 합니다...",
            "verdict": "보완필요",
            "score": 3,
            "reasoning": "기본적인 설계 원칙을 언급했지만, 상태 코드 처리와 버전 관리 등 실무적 관점이 부족합니다.",
            "recommended_answer": "REST API 설계 시에는 1) HTTP 메서드를 명확히 구분(GET/POST/PUT/DELETE)하고, 2) URI는 리소스 기반으로 설계하며..."
        }
    ],
    "overall_score": 4,
    "overall_feedback": "전반적으로 기본 개념 이해도가 좋으나 실무 경험에 기반한 심화 답변이 필요합니다.",
    "strengths": ["기본 개념에 대한 정확한 이해", "논리적인 답변 구조"],
    "improvements": ["실무 경험 기반 예시 추가", "심화 개념 학습 필요"],
    "model_used": "gemini-3-pro-preview",
    "debate_available": true
}
```

### Response Fields

| 필드 | 타입 | 설명 |
|------|------|------|
| success | boolean | 성공 여부 |
| session_id | string | 면접 세션 ID |
| questions | array | 각 질문별 분석 결과 |
| questions[].question | string | 면접 질문 |
| questions[].user_answer | string | 지원자 답변 |
| questions[].verdict | string | 판정: `적절`, `부적절`, `보완필요` |
| questions[].score | int | 점수 (1-5) |
| questions[].reasoning | string | 평가 근거 |
| questions[].recommended_answer | string \| null | 추천 답변 (보완/부적절 시 제공) |
| questions[].category | string | 질문 카테고리 |
| overall_score | int | 종합 점수 (0-5) |
| overall_feedback | string | 종합 피드백 |
| strengths | array | 강점 목록 |
| improvements | array | 개선점 목록 |
| model_used | string | 사용된 모델명 |
| debate_available | boolean | 심층 분석(토론) 가능 여부 |

### Error Responses

#### 400 Bad Request

**필수 필드 누락:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "session_id와 qa_pairs는 필수입니다"
    }
}
```

**빈 Q&A 목록:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "qa_pairs는 최소 1개 이상이어야 합니다"
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 500 Internal Server Error

**분석 실패:**
```json
{
    "detail": {
        "code": "ANALYSIS_FAILED",
        "message": "면접 분석 중 오류가 발생했습니다"
    }
}
```

#### 503 Service Unavailable

```json
{
    "detail": {
        "code": "LLM_UNAVAILABLE",
        "message": "AI 서비스에 연결할 수 없습니다"
    }
}
```

---

## 7. 심층 분석 - 토론 (2단계)

### 개요

사용자가 1단계 분석 결과를 보고 **심층 분석 버튼을 클릭**하면, GPT-4o가 동일 Q&A를 독립 분석한 뒤 Gemini 분석과 비교하여 토론을 진행하고 최종 합의 결과를 도출합니다.

> **트리거:** 사용자 수동 (심층 분석 버튼 클릭)
> **처리 흐름:** GPT-4o 독립 분석 → Gemini vs GPT-4o 비교 → 불일치 항목 토론 (1라운드) → 최종 합의
> **사용 모델:** `gemini-3-pro-preview` + `gpt-4o`

### Endpoint

```
POST /ai/evaluation/debate
```

### 처리 방식

**동기** - 즉시 응답 반환 (Gemini + OpenAI 호출로 20~60초 소요 가능)

### Request Headers

| Header | 값 | 필수 |
|--------|-----|:----:|
| X-API-Key | your-api-key-here | ✅ |
| Content-Type | application/json | ✅ |

### Request Body

```json
{
    "session_id": "session_abc123",
    "qa_pairs": [
        {
            "question": "Spring Boot에서 의존성 주입이 무엇인가요?",
            "answer": "의존성 주입은 객체가 필요로 하는 의존 객체를 외부에서 주입하는 패턴입니다...",
            "category": "cs_fundamentals"
        }
    ],
    "gemini_analysis": {
        "questions": [
            {
                "question": "Spring Boot에서 의존성 주입이 무엇인가요?",
                "user_answer": "...",
                "verdict": "적절",
                "score": 4,
                "reasoning": "...",
                "recommended_answer": null
            }
        ],
        "overall_score": 4,
        "overall_feedback": "...",
        "strengths": ["..."],
        "improvements": ["..."]
    },
    "resume_text": "3년 경력 백엔드 개발자...",
    "job_posting_text": "Spring Boot 경험 필수...",
    "interview_type": "tech"
}
```

### Request Fields

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| session_id | string | ✅ | 면접 세션 ID |
| qa_pairs | array | ✅ | 질의응답 목록 (1단계와 동일 형식) |
| gemini_analysis | object | ✅ | 1단계 Gemini 분석 결과 (전체 JSON) |
| resume_text | string | ❌ | 이력서 텍스트 (기본값: "") |
| job_posting_text | string | ❌ | 채용공고 텍스트 (기본값: "") |
| interview_type | string | ❌ | 면접 유형 (기본값: "tech") |

### Response (200 OK)

```json
{
    "success": true,
    "session_id": "session_abc123",
    "final_analysis": {
        "success": true,
        "session_id": "session_abc123",
        "questions": [
            {
                "question": "Spring Boot에서 의존성 주입이 무엇인가요?",
                "user_answer": "...",
                "verdict": "적절",
                "score": 4,
                "reasoning": "토론 합의: DI의 핵심 개념을 정확히 설명했으며...",
                "recommended_answer": null,
                "category": "cs_fundamentals"
            }
        ],
        "overall_score": 4,
        "overall_feedback": "토론 결과 종합 피드백...",
        "strengths": ["..."],
        "improvements": ["..."],
        "model_used": "gemini-3-pro-preview",
        "debate_available": false
    },
    "gemini_analysis": {
        "success": true,
        "session_id": "session_abc123",
        "questions": [ ... ],
        "overall_score": 4,
        "overall_feedback": "Gemini 분석 결과...",
        "model_used": "gemini-3-pro-preview",
        "debate_available": false
    },
    "gpt4o_analysis": {
        "success": true,
        "session_id": "session_abc123",
        "questions": [ ... ],
        "overall_score": 3,
        "overall_feedback": "GPT-4o 분석 결과...",
        "model_used": "gpt-4o",
        "debate_available": false
    },
    "disagreements": [
        {
            "question_index": 1,
            "question": "REST API 설계 시 주의할 점은?",
            "gemini_score": 4,
            "gpt4o_score": 2,
            "score_diff": 2
        }
    ],
    "consensus_method": "debated"
}
```

### Response Fields

| 필드 | 타입 | 설명 |
|------|------|------|
| success | boolean | 성공 여부 |
| session_id | string | 면접 세션 ID |
| final_analysis | object | 최종 분석 결과 (토론 합의) - AnalyzeInterviewResponse 형식 |
| gemini_analysis | object | Gemini 개별 분석 결과 - AnalyzeInterviewResponse 형식 |
| gpt4o_analysis | object \| null | GPT-4o 개별 분석 결과 (실패 시 null) |
| disagreements | array | 불일치 항목 목록 |
| disagreements[].question_index | int | 질문 인덱스 |
| disagreements[].question | string | 질문 내용 |
| disagreements[].gemini_score | int | Gemini 평가 점수 |
| disagreements[].gpt4o_score | int | GPT-4o 평가 점수 |
| disagreements[].score_diff | int | 점수 차이 |
| consensus_method | string | 합의 방법: `single` (GPT-4o 실패), `merged` (전부 일치), `debated` (토론 진행) |

### Consensus Method 설명

| Method | 설명 | 발생 조건 |
|--------|------|----------|
| `single` | Gemini 분석만 사용 | GPT-4o 분석 실패 시 |
| `merged` | 두 분석 단순 병합 | 모든 질문 점수 차이 < 2 |
| `debated` | 토론 후 합의 도출 | 점수 차이 ≥ 2인 질문 존재 시 |

### Error Responses

#### 400 Bad Request

**필수 필드 누락:**
```json
{
    "detail": {
        "code": "INVALID_REQUEST",
        "message": "session_id, qa_pairs, gemini_analysis는 필수입니다"
    }
}
```

#### 401 Unauthorized

```json
{
    "detail": {
        "code": "UNAUTHORIZED",
        "message": "유효하지 않은 API Key입니다"
    }
}
```

#### 500 Internal Server Error

**토론 실패:**
```json
{
    "detail": {
        "code": "DEBATE_FAILED",
        "message": "심층 분석 중 오류가 발생했습니다"
    }
}
```

#### 503 Service Unavailable

**토론 기능 비활성화:**
```json
{
    "detail": {
        "code": "DEBATE_UNAVAILABLE",
        "message": "토론 기능이 비활성화되어 있습니다. OpenAI API 키를 설정해주세요."
    }
}
```

**LLM 서비스 불가:**
```json
{
    "detail": {
        "code": "LLM_UNAVAILABLE",
        "message": "AI 서비스에 연결할 수 없습니다"
    }
}
```

---

## 공통 에러 응답

### HTTP Status Codes

| Status | 설명 | 사용 시점 |
|--------|------|----------|
| 400 | Bad Request | 필수 파라미터 누락, 잘못된 형식 |
| 401 | Unauthorized | X-API-Key 누락/불일치 |
| 404 | Not Found | task_id 없음, 파일 없음 |
| 410 | Gone | 작업 만료 |
| 422 | Unprocessable Entity | 파라미터 형식은 맞지만 처리 불가 |
| 429 | Too Many Requests | Rate Limit 초과 |
| 500 | Internal Server Error | 서버 내부 오류 |
| 503 | Service Unavailable | 외부 서비스(Gemini, S3) 연결 실패 |

### 에러 응답 형식

```json
{
    "detail": {
        "code": "ERROR_CODE",
        "message": "사람이 읽을 수 있는 에러 메시지",
        "field": "에러 발생 필드 (optional)"
    }
}
```

---

## 에러 코드 전체 목록

| Code | 설명 | 관련 API |
|------|------|----------|
| INVALID_REQUEST | 필수 파라미터 누락 | 전체 |
| INVALID_FILE_TYPE | 지원하지 않는 파일 타입 | 1, 4 |
| INVALID_DOCUMENT | 문서 정보 불완전 | 1 |
| INVALID_MODE | 잘못된 chat mode | 2 |
| INVALID_INTERVIEW_TYPE | 잘못된 면접 타입 | 2 |
| INVALID_TASK_ID | 잘못된 task_id 형식 | 5 |
| INVALID_URL | 잘못된 URL 형식 | 3, 4 |
| MISSING_CONTEXT | 필수 context 누락 | 2 |
| EMPTY_MESSAGE | 빈 메시지 | 2 |
| HISTORY_TOO_LONG | history 초과 | 2 |
| UNAUTHORIZED | 인증 실패 | 전체 |
| FILE_NOT_FOUND | 파일 없음 | 1, 2, 3, 4 |
| TASK_NOT_FOUND | 작업 없음 | 5 |
| TASK_EXPIRED | 작업 만료 | 5 |
| SESSION_NOT_FOUND | 면접 세션 없음 | 2 |
| OCR_FAILED | OCR 실패 | 1 |
| PARSE_FAILED | 파싱 실패 | 3 |
| NO_SCHEDULE_FOUND | 일정 없음 | 3 |
| MASKING_FAILED | 마스킹 실패 | 4 |
| ANALYSIS_FAILED | 면접 분석 실패 | 6 |
| DEBATE_FAILED | 심층 분석(토론) 실패 | 7 |
| DEBATE_UNAVAILABLE | 토론 기능 비활성화 (OpenAI 키 미설정) | 7 |
| STREAM_ERROR | 스트리밍 오류 | 2 |
| RATE_LIMIT_EXCEEDED | 요청 한도 초과 | 전체 |
| INTERNAL_ERROR | 내부 서버 오류 | 전체 |
| LLM_UNAVAILABLE | LLM 서비스 불가 | 1, 2, 3, 6, 7 |
| S3_UNAVAILABLE | S3 서비스 불가 | 1, 4 |
| VECTORDB_UNAVAILABLE | VectorDB 서비스 불가 | 2 |

---

## Rate Limiting

| API 유형 | 제한 |
|----------|------|
| 동기 API | 100 requests/min |
| 비동기 API | 50 requests/min |
| 스트리밍 API | 20 connections/min |

### 헤더 응답

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## 백엔드 ↔ AI 서버 매핑 가이드

백엔드 외부 API와 AI 서버 내부 API의 매핑:

| 백엔드 외부 API | AI 서버 내부 API | 설명 |
|----------------|------------------|------|
| `POST /api/ai-chatrooms/{roomId}/messages` | `POST /ai/chat` (mode: general) | 일반 채팅 |
| `POST /api/ai/chatrooms/{roomId}/analysis` | `POST /ai/text/extract` | 분석 요청 |
| `POST /api/ai/chatrooms/{roomId}/interview` | `POST /ai/chat` (mode: interview_question) | 면접 시작 |
| `POST /api/ai/chatrooms/{roomId}/evaluation` | `POST /ai/chat` (mode: interview_report) | 면접 평가 |
| `POST /api/ai/events/extraction` | `POST /ai/calendar/parse` | 일정 추출 |
| `POST /api/board/upload` | `POST /ai/masking/draft` | 마스킹 요청 |
| `POST /api/ai/chatrooms/{roomId}/evaluation/analyze` | `POST /ai/evaluation/analyze` | 면접 답변 분석 (1단계) |
| `POST /api/ai/chatrooms/{roomId}/evaluation/debate` | `POST /ai/evaluation/debate` | 심층 분석 - 토론 (2단계) |

---

## 에러 처리 권장 사항

### Backend 에러 핸들링

```java
// Spring Boot 예시
@ExceptionHandler
public ResponseEntity<ErrorResponse> handleAIServerError(AIServerException e) {
    switch (e.getCode()) {
        case "TASK_NOT_FOUND":
            return ResponseEntity.status(404).body(e.toResponse());
        case "LLM_UNAVAILABLE":
            // 재시도 로직 또는 사용자에게 안내
            return ResponseEntity.status(503).body(e.toResponse());
        default:
            return ResponseEntity.status(500).body(e.toResponse());
    }
}
```

### 재시도 정책

| 에러 코드 | 재시도 | 설명 |
|----------|:------:|------|
| LLM_UNAVAILABLE | ✅ | 최대 3회, 지수 백오프 (1초 → 2초 → 4초) |
| S3_UNAVAILABLE | ✅ | 최대 3회, 지수 백오프 |
| VECTORDB_UNAVAILABLE | ✅ | 최대 3회, 지수 백오프 |
| RATE_LIMIT_EXCEEDED | ✅ | X-RateLimit-Reset 헤더 확인 후 재시도 |
| TASK_EXPIRED | ❌ | 새 요청 필요 |
| OCR_FAILED | ❌ | 파일 확인 필요 |
| UNAUTHORIZED | ❌ | API Key 확인 필요 |

---

**문서 작성:** 2026-01-13
**작성자:** AI Team
**버전:** v1.9
**마지막 업데이트:** 2026-02-08

### 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.9 | 2026-02-08 | 면접 답변 평가 API 추가: 6번 면접 답변 분석 (Gemini 3 Pro, 1단계), 7번 심층 분석 토론 (Gemini×GPT-4o, 2단계 사용자 수동 트리거) |
| v1.8 | 2026-02-08 | 최종 API 명세 확정: 텍스트 추출 결과에 summary 필드 추가, 채팅 404/500 에러 코드 상세화 (VECTORDB_ERROR, PARSE_FAILED, INTERNAL_ERROR, LLM_ERROR) |
| v1.7 | 2026-01-27 | 백엔드 회의 후 API 명세 업데이트: s3_key 필드 통일, 면접 리포트 context 배열 구조, 에러 응답 상세화 |
| v1.5 | 2026-01-25 | API 명세 최신화: file_url 필드 추가, 에러 응답 상세화, Request/Response 예시 보완 |
| v1.4 | 2026-01-23 | API별 상세 에러 케이스 추가 |
| v1.3 | 2026-01-23 | `/ai/text/extract`를 batch 구조로 변경 (resume + job_posting 동시 처리) |
| v1.2 | 2026-01-23 | Request Body 필드명 RDB 컬럼명과 동기화 |
| v1.1 | 2026-01-23 | 초기 API 명세 작성 |
