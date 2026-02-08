# 9팀 Devths API 명세서

## 목차
1. [FASTAPI EndPoint (AI 서버)](#1-fastapi-endpoint-ai-서버)
2. [공통 (Common)](#2-공통-common)
3. [AI 분석](#3-ai-분석)
4. [AI 채팅방](#4-ai-채팅방-ai-chats)
5. [인증 (Auth)](#5-인증-auth)
6. [회원 (Users)](#6-회원-users)
7. [캘린더 (Calendars)](#7-캘린더-calendars)
8. [게시판 (Posts)](#8-게시판-posts)
9. [댓글 (Comments)](#9-댓글-comments)
10. [알림 (Notifications)](#10-알림-notifications)
11. [실시간 채팅 (Chats)](#11-실시간-채팅-chats)

---

## 1. FASTAPI EndPoint (AI 서버)

> **Backend → AI Server 내부 통신 API**
>
> Base URL: `http://ai-server:8000`

### 1.1 텍스트 추출 + 임베딩
- **Method**: `POST`
- **URL**: `/ai/text/extract`
- **Header**: `X-API-Key`, `Content-Type: application/json`
- **Request Body**:
```json
{
    "model": "GEMINI",
    "resume": {
        "user_id": 12,
        "file_id": 23,
        "s3_key": "resumes/2026/01/uuid_resume.pdf",
        "file_type": "image/png",
        "text": null
    },
    "jobPost": {
        "user_id": 12,
        "file_id": 24,
        "s3_key": "job_posts/2026/01/uuid_posting.png",
        "file_type": "image/png",
        "text": null
    }
}
```
- **Request Fields**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| model | String | ✅ | `GEMINI`, `VLLM` |
| resume | Object | ⚠️ | 이력서 정보 (resume 또는 jobPost 중 하나 필수) |
| resume.user_id | Long | ✅ | 사용자 ID |
| resume.file_id | Long | ⚠️ | s3_attachments 테이블 ID |
| resume.s3_key | String | ⚠️ | S3 파일 경로 |
| resume.file_type | String | ⚠️ | MIME 타입 (image/png, image/jpeg, application/pdf) |
| resume.text | String | ⚠️ | 직접 입력 텍스트 (file_id/s3_key와 택1) |
| jobPost | Object | ⚠️ | 채용공고 정보 (resume와 동일 구조) |

- **Response Status Code**: `202`
- **Response**:
```json
{
    "task_id": "task_abc123def456",
    "status": "processing"
}
```

### 1.2 텍스트 추출 + 임베딩 (Polling 완료시)
- **Method**: `GET`
- **URL**: `/ai/task/{task_id}`
- **Response Status Code**: `200`
- **Response**:
```json
{
    "task_id": "task_abc123def456",
    "status": "completed",
    "result": {
        "success": true,
        "extracted_text": "추출된 전체 텍스트...",
        "pages": [
            {"page": 1, "text": "1페이지 텍스트..."},
            {"page": 2, "text": "2페이지 텍스트..."}
        ]
    }
}
```

### 1.3 채팅
- **Method**: `POST`
- **URL**: `/ai/chat`
- **Header**: `X-API-Key`, `Content-Type: application/json`
- **Request Body**:
```json
{
    "model": "gemini",
    "room_id": "room_001",
    "user_id": "user_456",
    "message": "이력서 작성 팁 알려줘",
    "context": {
        "mode": "general",
        "resume": {
            "file_id": 23,
            "s3_key": "resumes/2026/01/uuid_resume.pdf",
            "file_type": "application/pdf",
            "text": null
        },
        "jobPost": {
            "file_id": 24,
            "s3_key": "job_posts/2026/01/uuid_posting.png",
            "file_type": "image/png",
            "text": null
        },
        "interview_type": null,
        "session_id": null,
        "question_count": null
    },
    "history": [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕하세요!"}
    ]
}
```
- **Request Fields**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| model | String | ✅ | `gemini`, `openai`, `vllm` |
| room_id | String | ✅ | 채팅방 ID |
| user_id | String | ✅ | 사용자 ID |
| message | String | ⚠️ | 사용자 메시지 |
| context.mode | String | ✅ | `general`, `analysis`, `interview_question`, `interview_report` |
| context.resume | Object | ❌ | 이력서 정보 (분석 시) |
| context.jobPost | Object | ❌ | 채용공고 정보 (분석 시) |
| context.interview_type | String | ⚠️ | `technical`, `personality` (면접 모드 시) |
| context.session_id | String | ⚠️ | 면접 세션 ID (면접 모드 시) |
| context.question_count | Number | ❌ | 현재까지 생성된 질문 수 |
| history | Array | ❌ | 대화 히스토리 (최대 20개) |

- **Response Status Code**: `200` (SSE Streaming)
- **Response**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "general",
    "response": "이력서 작성 팁을 알려드릴게요...",
    "data": {
        "analysis": null,
        "interview": null,
        "report": null
    },
    "metadata": {
        "tool_used": "RAG",
        "description": "VectorDB 검색 후 LLM 응답 생성",
        "processing_time_ms": 1234,
        "tokens_used": 500
    }
}
```

**분석 모드 응답 (mode: analysis)**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "analysis",
    "response": "이력서와 채용공고를 분석했습니다.",
    "data": {
        "analysis": {
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
            },
            "matching": {
                "score": 75,
                "grade": "B",
                "matched_skills": ["협업", "문제해결"],
                "missing_skills": ["Java", "Spring"]
            }
        },
        "interview": null,
        "report": null
    },
    "metadata": {
        "tool_used": "RAG",
        "description": "이력서/채용공고 분석",
        "processing_time_ms": 2500,
        "tokens_used": 800
    }
}
```

**면접 질문 모드 응답 (mode: interview_question)**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "interview_question",
    "response": null,
    "data": {
        "analysis": null,
        "interview": {
            "type": "technical",
            "question": "React와 Vue의 차이점에 대해 설명해주세요.",
            "question_number": 1
        },
        "report": null
    },
    "metadata": {
        "tool_used": "LLM",
        "description": "면접 질문 생성",
        "processing_time_ms": 1100,
        "tokens_used": 300
    }
}
```

**면접 리포트 모드 응답 (mode: interview_report)**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "interview_report",
    "response": "면접 평가가 완료되었습니다.",
    "data": {
        "analysis": null,
        "interview": null,
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
    },
    "metadata": {
        "tool_used": "LLM",
        "description": "면접 평가 리포트 생성",
        "processing_time_ms": 3200,
        "tokens_used": 1200
    }
}
```

### 1.4 캘린더 일정 파싱
- **Method**: `POST`
- **URL**: `/ai/calendar/parse`
- **Header**: `X-API-Key`, `Content-Type: application/json`
- **Request Body**:
```json
{
    "model": "gemini",
    "user_id": "user_456",
    "source": {
        "file_id": 23,
        "s3_key": "job_posts/2026/01/uuid_posting.png",
        "file_type": "image/png",
        "text": null
    }
}
```
- **Response Status Code**: `200`
- **Response**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "calendar_parsing",
    "response": "채용 일정을 파싱했습니다.",
    "data": {
        "calendar": {
            "company": "카카오",
            "position": "백엔드 개발자",
            "schedules": [
                {
                    "stage": "서류 마감",
                    "date": "2026-01-15",
                    "time": null
                },
                {
                    "stage": "코딩테스트",
                    "date": "2026-01-20",
                    "time": "14:00"
                },
                {
                    "stage": "1차 면접",
                    "date": "2026-01-25",
                    "time": null
                }
            ],
            "hashtags": ["#카카오", "#백엔드", "#신입"],
            "total_count": 3
        }
    },
    "metadata": {
        "tool_used": "OCR+LLM",
        "description": "채용공고에서 일정 정보 추출",
        "processing_time_ms": 1850,
        "tokens_used": 420
    }
}
```

### 1.5 게시판 첨부파일 마스킹
- **Method**: `POST`
- **URL**: `/ai/masking/draft`
- **Header**: `X-API-Key`, `Content-Type: application/json`
- **Request Body**:
```json
{
    "model": "gemini",
    "user_id": "user_456",
    "context": {
        "mode": "masking",
        "source": {
            "file_id": 23,
            "s3_key": "posts/2026/01/uuid_document.png",
            "file_url": "https://s3.../document.png",
            "file_type": "image/png"
        }
    }
}
```
- **Response Status Code**: `202`
- **Response**:
```json
{
    "success": true,
    "task_id": "task_masking_001",
    "status": "processing",
    "polling_url": "/ai/masking/status/task_masking_001"
}
```

### 1.6 게시판 첨부파일 마스킹 (Polling 완료시)
- **Method**: `GET`
- **URL**: `/ai/masking/status/{task_id}`
- **Response Status Code**: `202` (처리중) / `200` (완료)

**처리중 Response**:
```json
{
    "success": true,
    "task_id": "task_masking_001",
    "status": "processing",
    "progress": 65,
    "message": "OCR 처리 중...",
    "estimated_time_remaining_ms": 5000
}
```

**완료 Response**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "masking",
    "task_id": "task_masking_001",
    "status": "completed",
    "response": "개인정보 마스킹이 완료되었습니다.",
    "data": {
        "masking": {
            "original_url": "https://s3.../document.png",
            "masked_url": "https://s3.../document_masked.png",
            "thumbnail_url": "https://s3.../document_masked_thumb.png",
            "detected_pii": [
                {
                    "type": "name",
                    "coordinates": [100, 50, 200, 80],
                    "confidence": 0.95
                },
                {
                    "type": "phone",
                    "coordinates": [100, 100, 250, 130],
                    "confidence": 0.92
                },
                {
                    "type": "email",
                    "coordinates": [100, 150, 300, 180],
                    "confidence": 0.98
                },
                {
                    "type": "face",
                    "coordinates": [400, 50, 500, 150],
                    "confidence": 0.99
                }
            ],
            "total_masked": 4
        }
    },
    "metadata": {
        "tool_used": "OCR+NER+FaceDetection+Masking",
        "description": "개인정보 탐지 및 마스킹 처리 완료",
        "processing_time_ms": 3200,
        "tokens_used": 0
    }
}
```

### 1.7 비동기 작업 상태 조회
- **Method**: `GET`
- **URL**: `/ai/task/{task_id}`
- **Header**: `X-API-Key`

**처리중 Response (Status: 200)**:
```json
{
    "success": true,
    "task_id": "task_abc123",
    "status": "processing",
    "progress": 65,
    "message": "OCR 처리 중...",
    "estimated_time_remaining_ms": 5000
}
```

**완료 Response (Status: 200)**:
```json
{
    "success": true,
    "model": "gemini",
    "mode": "masking",
    "task_id": "task_abc123",
    "status": "completed",
    "response": "작업이 완료되었습니다.",
    "data": {
        "masking": {
            "original_url": "https://s3.../document.png",
            "masked_url": "https://s3.../document_masked.png",
            "detected_pii": [...],
            "total_masked": 4
        }
    },
    "metadata": {
        "tool_used": "OCR+NER+Masking",
        "description": "개인정보 마스킹 완료",
        "processing_time_ms": 3200,
        "tokens_used": 0
    }
}
```

**실패 Response (Status: 200)**:
```json
{
    "success": false,
    "task_id": "task_abc123",
    "status": "failed",
    "error": {
        "code": "OCR_ERROR",
        "message": "파일 형식을 인식할 수 없습니다.",
        "details": "Supported formats: image/png, image/jpeg, application/pdf"
    },
    "metadata": {
        "processing_time_ms": 1200,
        "failed_at": "2026-01-23T16:32:00+09:00"
    }
}
```

---

## 2. 공통 (Common)

### 2.1 Presigned URL 발급
- **Method**: `POST`
- **URL**: `/api/files/presigned`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "fileName": "my_resume.pdf",
    "mimeType": "application/pdf"
}
```
- **설명**: 사용자가 AI 챗봇, 게시글, 실시간 채팅에서 파일을 첨부하는 과정에서 호출되는 Presigned URL 발급 API
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "Presigned URL이 생성되었습니다.",
    "data": {
        "presignedUrl": "https://s3.region.amazonaws.com/bucket/...",
        "s3Key": "resumes/2026/01/uuid_my_resume.pdf"
    },
    "timestamp": "2026-01-12T13:20:10.123"
}
```

### 2.2 S3 파일 첨부
- **Method**: `POST`
- **URL**: `/api/files`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "originalName": "my_resume.pdf",
    "s3Key": "resumes/2026/01/uuid_my_resume.pdf",
    "mimeType": "application/pdf",
    "category": "RESUME",
    "fileSize": 1048576,
    "refType": "CHATROOM",
    "refId": 123,
    "sortOrder": 1
}
```
- **설명**:
  - 발급한 Presigned URL을 통해 프론트엔드 측에서 S3에 파일을 업로드
  - 해당 파일의 S3 Key와 기타 데이터를 받아 저장하는 API
  - URL 발급과 첨부 프로세스의 분리를 통해 서버를 거치지 않고 FE에서 S3로 직접 업로드
  - 대용량 파일 처리 시 서버 리소스 보호
  - 여러 개의 파일을 병렬로 업로드하는 요구사항이 존재, 해당 상황에서 유리
- **Response Status Code**: `201`, `400`, `401`, `500`

### 2.3 S3 파일 삭제
- **Method**: `DELETE`
- **URL**: `/api/files/{fileId}`
- **Header**: `Authorization: Bearer {accessToken}`
- **필수 파라미터**: `fileId: Long`
- **Response Status Code**: `204`, `400`, `401`, `403`, `404`, `500`

### 2.4 비동기 작업 상태 확인
- **Method**: `GET`
- **URL**: `/api/ai/tasks/{taskId}`
- **Header**: `Authorization: Bearer {accessToken}`
- **필수 파라미터**: `taskId: Long`
- **설명**: 이미지/PDF 분석 등 시간이 오래 걸리는 AI 작업을 비동기로 처리하고 작업 진행 상황을 서버에 요청하는 API. 초기 구현 비용의 절감 및 인프라 구성의 단순화를 위해 폴링 방식 채택.
- **Response Status Code**: `200`
- **Response**:
```json
{
    "message": "이력서 및 포트폴리오 분석에 성공하였습니다.",
    "data": {
        "taskId": 123,
        "taskType": "RESUME",
        "referenceId": 45,
        "status": "COMPLETED",
        "result": {
            "roomId": 52,
            "messageId": 501,
            "interviewId": null,
            "role": "ASSISTANT",
            "type": "REPORT",
            "content": "## 분석 결과 리포트\\n기타의 역량은...",
            "metadata": {
                "score": 95,
                "summary": "신입 FE 개발자 역량 분석 리포트",
                "strengths": ["React", "TypeScript"]
            },
            "createdAt": "2026-01-12T11:04:53"
        },
        "createdAt": "2026-01-12T10:41:33",
        "updatedAt": "2026-01-12T11:04:53",
        "isNotified": true
    },
    "timestamp": "2026-01-12T13:20:10.123"
}
```

---

## 3. AI 분석

### 3.1 이력서/포트폴리오 분석
- **Method**: `POST`
- **URL**: `/api/ai/chatrooms/{roomId}/analysis`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "resumeId": 10,
    "portfolioId": 11,
    "jobPostingId": 12
}
```
- **필수 파라미터**: `roomId: Long`
- **선택 파라미터**: `resumeId: Long`, `portfolioId: Long`, `jobPostingId: Long`
- **설명**:
  - 사용자가 첨부한 이력서/포트폴리오 및 채용 공고를 바탕으로 분석을 요청하는 API
  - 분석이 이미 진행 중이라면 새로운 분석을 실행하지 않고 기존 작업의 taskId를 반환하거나 진행 상태를 알려주는 방식으로 멱등성을 보장
- **Response Status Code**: `202`, `400`, `401`, `403`, `404`, `500`
- **Response (202)**:
```json
{
    "message": "이력서 및 포트폴리오 분석이 시작되었습니다.",
    "data": {
        "taskId": 501,
        "status": "PENDING"
    },
    "timestamp": "2026-01-12T13:20:10.123"
}
```

### 3.2 면접 모드 시작
- **Method**: `POST`
- **URL**: `/api/ai/chatrooms/{roomId}/interview`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "interviewType": "TECH"
}
```
- **필수 파라미터**: `roomId: Long`
- **선택 파라미터**: `interviewType: String`
- **설명**: 사용자가 첨부한 이력서/포트폴리오를 기반으로 모의 면접을 시작할 수 있는 API
- **Response Status Code**: `200`, `400`, `401`, `403`, `500`
- **Response (200)**:
```json
{
    "message": "면접 모드가 시작되었습니다.",
    "data": {
        "interviewId": 1,
        "status": "IN_PROGRESS",
        "content": {
            "messageId": 100,
            "role": "ASSISTANT",
            "content": "기술 모의 면접을 시작하겠습니다. 첫 번째 질문으로...",
            "type": "INTERVIEW",
            "createdAt": "2026-01-12T11:30:00"
        }
    },
    "timestamp": "2026-01-12T13:20:10.123"
}
```

---

## 4. AI 채팅방 (AI Chats)

### 4.1 AI 채팅방 목록 조회
- **Method**: `GET`
- **URL**: `/api/ai-chatrooms?size=n&lastId=k`
- **Header**: `Authorization: Bearer {accessToken}`
- **선택 파라미터**:
  - `size: Int` (default 10)
  - `lastId: Long`
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "AI 채팅방 목록을 성공적으로 조회하였습니다.",
    "data": {
        "rooms": [
            {
                "roomId": 1,
                "roomUuid": "550e8400-e29b-41d4-a716-446655440000",
                "title": "자바 백엔드 기술 면접 준비",
                "createdAt": "2026-01-12T10:00:00",
                "updatedAt": "2026-01-12T15:30:00"
            }
        ],
        "lastId": 10,
        "hasNext": true
    },
    "timestamp": "2026-01-12T18:55:00.000"
}
```

### 4.2 AI 채팅방 생성
- **Method**: `POST`
- **URL**: `/api/ai-chatrooms`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**: `title` (기본값은 "새 채팅방")
- **Response Status Code**: `201`, `400`, `401`, `500`
- **Response (201)**:
```json
{
    "message": "AI 채팅방이 성공적으로 생성되었습니다.",
    "data": {
        "roomId": 1,
        "roomUuid": "550e8400-e29b-41d4-a716-446655440000",
        "title": "새 채팅방",
        "createdAt": "2026-01-12T19:10:00"
    },
    "timestamp": "2026-01-12T19:10:00.000"
}
```

---

## 5. 인증 (Auth)

### 5.1 구글 소셜 로그인
- **Method**: `POST`
- **URL**: `/api/auth/google`
- **Request Body**:
```json
{
    "authCode": "4/0AfgeXv..."
}
```
- **필수 파라미터**: `authCode: String`
- **설명**:
  - OAuth2 기반 구글 소셜 로그인 요청 API
  - 사용자 정보를 담고 있는 idToken이 아닌, 토큰 발급용 임시 문자열인 authCode를 사용하여 Server-Side Flow로 인증을 처리
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response Header (200)**:
```
Authorization: Bearer {accessToken}
Set-Cookie: refreshToken={refreshToken}; HttpOnly; Secure; SameSite=...
Access-Control-Expose-Headers: Authorization
Access-Control-Allow-Credentials: true
```

### 5.2 토큰 재발급
- **Method**: `POST`
- **URL**: `/api/auth/tokens`
- **Header**: `Cookie: refreshToken`
- **설명**: 사용자의 RT를 기반으로 새로운 AT를 발급해주는 API. Refresh Token Rotation(RTR)을 통해 보안 강화
- **Response Status Code**: `200`, `400`, `401`, `403`, `500`
- **Response Header (200)**:
```
Authorization: Bearer {newAccessToken}
Set-Cookie: refreshToken={newRefreshToken}; HttpOnly; Secure; SameSite=...
```
- **Response (204)**: 로그아웃 처리 시
```
Set-Cookie: refreshToken=; Max-Age=0; HttpOnly; Secure; SameSite=...
```

---

## 6. 회원 (Users)

### 6.1 내 정보 조회
- **Method**: `GET`
- **URL**: `/api/users/me`
- **Header**: `Authorization: Bearer {accessToken}`
- **Response Status Code**: `200`, `400`, `401`, `500`

### 6.2 내 정보 수정
- **Method**: `PUT`
- **URL**: `/api/users/me`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "nickname": "새로운닉네임",
    "interests": ["프론트엔드", "React"]
}
```
- **필수 파라미터**: `nickname: String`
- **선택 파라미터**: `interests: String[]`
- **설명**:
  - 클라이언트가 현재 화면에 보이는 모든 정보를 한 번에 보내고, 서버는 전달받은 값 그대로 리소스를 교체
  - Http Method로 PUT 사용
  - 필수 값인 nickname에 대해서는 백엔드 로직에서 Validation
- **Response Status Code**: `200`, `400`, `401`, `500`

---

## 7. 캘린더 (Calendars)

### 7.1 일정 추가
- **Method**: `POST`
- **URL**: `/api/events`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "step": "INTERVIEW",
    "title": "카카오 1차 면접",
    "company": "카카오",
    "startTime": "2026-01-20T14:00:00",
    "endTime": "2026-01-20T15:00:00",
    "description": "판교어딘 입렵",
    "tags": ["카카오", "대기업"],
    "notificationTime": 3,
    "notificationUnit": "DAY"
}
```
- **필수 파라미터**: `step: String`, `title: String`, `company: String`, `startTime: Datetime`, `endTime: Datetime`, `notificationTime: Int`, `notificationUnit: String`
- **선택 파라미터**: `description: String`, `tags: String[]`
- **Response Status Code**: `201`, `400`, `401`, `500`
- **Response (201)**:
```json
{
    "message": "일정이 성공적으로 추가되었습니다.",
    "data": {
        "eventId": "google_event_id_xyz123"
    },
    "timestamp": "2026-01-12T16:55:00.00"
}
```

### 7.2 일정 목록 조회
- **Method**: `GET`
- **URL**: `/api/events?step=String&tag=String&startDate=yyyy-mm-dd&endDate=yyyy-mm-dd`
- **Header**: `Authorization: Bearer {accessToken}`
- **필수 파라미터**: `startDate: yyyy-mm-dd`, `endDate: yyyy-mm-dd`
- **선택 파라미터**: `step: String`, `tag: String`
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "일정 목록을 성공적으로 조회했습니다.",
    "data": [
        {
            "eventId": "v7n9m...",
            "title": "카카오 1차 면접",
            "startTime": "2026-01-20T14:00:00",
            "endTime": "2026-01-20T15:00:00",
            "step": "INTERVIEW",
            "tags": ["카카오", "대기업"]
        }
    ],
    "timestamp": "2026-01-12T17:30:00.00"
}
```

---

## 8. 게시판 (Posts)

### 8.1 게시글 목록 조회
- **Method**: `GET`
- **URL**: `/api/posts?size=n&lastId=k&tag=String`
- **Header**: `Authorization: Bearer {accessToken}`
- **선택 파라미터**:
  - `size: Int` (default = 20)
  - `lastId: Long`
  - `tag: String`
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "게시글 목록을 성공적으로 조회하였습니다.",
    "data": {
        "posts": [
            {
                "postId": 125,
                "title": "카카오 백엔드 면접 후기 공유합니다",
                "previewContent": "오늘 카카오 공채 백엔드 면접을 보고 왔습니다. 분위기는 생각보다...",
                "user": {
                    "userId": 47,
                    "nickname": "yun",
                    "profileImage": "https://...",
                    "interests": ["백엔드"]
                },
                "likeCount": 12,
                "commentCount": 5,
                "shareCount": 8,
                "tags": ["면접"],
                "createdAt": "2026-01-12T15:00:00"
            }
        ],
        "lastId": 106,
        "hasNext": true
    },
    "timestamp": "2026-01-12T16:30:00.000"
}
```

### 8.2 게시글 상세 조회
- **Method**: `GET`
- **URL**: `/api/posts/{postId}`
- **Header**: `Authorization: Bearer {accessToken}`
- **필수 파라미터**: `postId: Long`
- **Response Status Code**: `200`, `400`, `401`, `404`, `500`
- **Response (200)**:
```json
{
    "message": "게시글을 성공적으로 조회하였습니다.",
    "data": {
        "postId": 125,
        "title": "카카오 백엔드 면접 후기 공유합니다",
        "content": "오늘 카카오 공채 백엔드 면접을 보고 왔습니다. 분위기는 생각보다...",
        "attachments": [
            {
                "fileId": 1,
                "fileUrl": "https://s3.ap-northeast-2.../image1.jpg",
                "fileName": "스크린샷.jpg",
                "fileSize": 102400,
                "fileType": "IMAGE",
                "sortOrder": 0
            }
        ],
        "user": {
            "userId": 47,
            "nickname": "yun",
            "profileImage": "https://...",
            "interests": ["백엔드"]
        },
        "likeCount": 12,
        "commentCount": 5,
        "shareCount": 8,
        "tags": ["면접"],
        "createdAt": "2026-01-12T15:00:00",
        "updatedAt": "2026-01-12T15:00:00",
        "isLiked": true
    },
    "timestamp": "2026-01-12T16:30:00.000"
}
```

### 8.3 게시글 작성
- **Method**: `POST`
- **URL**: `/api/posts`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "title": "카카오 면접 준비",
    "content": "카카오 면접을 준비하면서...",
    "tags": ["면접"],
    "fileIds": [1, 5, 7]
}
```
- **필수 파라미터**: `title: String`, `content: String`
- **선택 파라미터**: `tags: String[]`, `fileIds: Long[]`
- **Response Status Code**: `201`, `400`, `401`, `500`
- **Response (201)**:
```json
{
    "message": "게시글이 성공적으로 등록되었습니다.",
    "data": {"postId": 125},
    "timestamp": "2026-01-12T17:10:00.000"
}
```

### 8.4 게시글 수정
- **Method**: `PUT`
- **URL**: `/api/posts/{postId}`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "title": "카카오 면접 준비",
    "content": "카카오 면접을 준비하면서...",
    "tags": ["면접"],
    "fileIds": [1, 5, 7]
}
```
- **필수 파라미터**: `postId: Long`, `title: String`, `content: String`
- **선택 파라미터**: `tags: String[]`, `fileIds: Long[]`
- **Response Status Code**: `200`, `400`, `401`, `403`, `500`
- **Response (200)**:
```json
{
    "message": "게시글이 성공적으로 수정되었습니다.",
    "data": {"postId": 125},
    "timestamp": "2026-01-12T17:10:00.000"
}
```

---

## 9. 댓글 (Comments)

### 9.1 댓글 목록 조회
- **Method**: `GET`
- **URL**: `/api/posts/{postId}/comments?size=n&lastId=k`
- **Header**: `Authorization: Bearer {accessToken}`
- **필수 파라미터**: `postId: Long`
- **선택 파라미터**:
  - `size: Int` (default 10)
  - `lastId: Long`
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "댓글 목록을 성공적으로 조회하였습니다.",
    "data": {
        "comments": [
            {
                "commentId": 11,
                "parentId": null,
                "content": "댓글",
                "user": {
                    "userId": 10,
                    "nickname": "yun",
                    "profileImage": "https://..."
                },
                "createdAt": "2026-01-12T17:00:00",
                "isDeleted": false
            },
            {
                "commentId": 12,
                "parentId": 11,
                "content": "대댓글",
                "user": {
                    "userId": 13,
                    "nickname": "cheol",
                    "profileImage": "https://..."
                },
                "createdAt": "2026-01-12T17:00:00",
                "isDeleted": false
            }
        ],
        "lastId": 35,
        "hasNext": true
    },
    "timestamp": "2026-01-12T18:30:00.000"
}
```

### 9.2 댓글 등록
- **Method**: `POST`
- **URL**: `/api/posts/{postId}/comments`
- **필수 파라미터**: `postId: Long`, `content: String`
- **선택 파라미터**: `parentId: Long` (대댓글의 경우)
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "댓글이 성공적으로 등록되었습니다.",
    "data": {"commentId": 3},
    "timestamp": "2026-01-12T18:00:00.000"
}
```

---

## 10. 알림 (Notifications)

### 10.1 알림 목록 조회
- **Method**: `GET`
- **URL**: `/api/notifications?size=n&lastId=k`
- **Header**: `Authorization: Bearer {accessToken}`
- **선택 파라미터**:
  - `size: Int` (default 10)
  - `lastId: Long`
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "알림 목록을 성공적으로 조회하였습니다.",
    "data": {
        "notifications": [
            {
                "notificationId": 1,
                "sender": {"senderId": 321, "senderName": "cheol"},
                "category": "POST",
                "type": "COMMENT",
                "content": "새로운 댓글이 달렸습니다.",
                "targetPath": "/posts/57",
                "resourceId": 57,
                "createdAt": "2026-01-12T10:00:00",
                "isRead": true
            }
        ],
        "lastId": 10,
        "hasNext": true
    },
    "timestamp": "2026-01-12T18:55:00.000"
}
```

### 10.2 푸시 토큰 등록
- **Method**: `POST`
- **URL**: `/api/notifications/tokens/{deviceId}`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "token": "fcm_token_value_12345...",
    "deviceType": "ANDROID"
}
```
- **필수 파라미터**: `deviceId: String`, `token: String`, `deviceType: String` (ANDROID, IOS, WEB)
- **Response Status Code**: `201`, `400`, `401`, `500`
- **Response (201)**:
```json
{
    "message": "푸시 토큰이 성공적으로 등록되었습니다.",
    "data": {
        "tokenId": 1,
        "createdAt": "2026-01-12T20:00:00"
    },
    "timestamp": "2026-01-12T20:00:00.000"
}
```

---

## 11. 실시간 채팅 (Chats)

### 11.1 참여 채팅방 목록 조회
- **Method**: `GET`
- **URL**: `/api/chatrooms?size=n&cursor=k&type={type}`
- **Header**: `Authorization: Bearer {accessToken}`
- **필수 파라미터**: `type: String` (default PRIVATE)
- **선택 파라미터**:
  - `size: Int` (default 10)
  - `cursor: DATETIME(개인/그룹) 또는 Int(인기)`
- **Response Status Code**: `200`, `400`, `401`, `500`
- **Response (200)**:
```json
{
    "message": "채팅방 목록을 성공적으로 조회하였습니다.",
    "data": {
        "chatRooms": [
            {
                "roomId": 101,
                "title": "yun",
                "lastMessageContent": "오늘 점심 메뉴 뭐야?",
                "lastMessageAt": "2026-01-12T14:00:00",
                "currentCount": 2,
                "tag": null
            },
            {
                "roomId": 98,
                "title": "cheol",
                "lastMessageContent": "사진을 보냈습니다.",
                "lastMessageAt": "2026-01-12T13:45:20",
                "currentCount": 2,
                "tag": null
            }
        ],
        "cursor": "2026-01-12T13:45:20",
        "hasNext": true
    },
    "timestamp": "2026-01-12T10:46:20.932"
}
```

### 11.2 새로운 채팅방 생성
- **Method**: `POST`
- **URL**: `/api/chatrooms`
- **Header**: `Authorization: Bearer {accessToken}`
- **Request Body**:
```json
{
    "type": "PRIVATE",
    "title": null,
    "tag": null,
    "userIds": [63]
}
```
- **필수 파라미터**: `type: String`, `userIds: Long[]`
- **선택 파라미터**: `title: String`, `tag: String`
- **Response Status Code**: `201`, `400`, `401`, `500`
- **Response (201)**:
```json
{
    "message": "채팅방이 성공적으로 생성되었습니다.",
    "data": {
        "roomId": 125,
        "type": "PRIVATE",
        "title": "yun",
        "inviteCode": "AX8291KL",
        "createdAt": "2026-01-12T20:55:10"
    },
    "timestamp": "2026-01-12T20:55:10.432"
}
```

---

## 공통 에러 응답 형식

모든 API의 에러 응답은 다음 형식을 따릅니다:
```json
{
    "message": "에러 메시지",
    "data": null,
    "timestamp": "2026-01-12T10:46:20.932"
}
```

### 공통 HTTP Status Code
| Code | 설명 |
|------|------|
| 200 | 성공 |
| 201 | 생성 성공 |
| 202 | 비동기 작업 시작 |
| 204 | 성공 (응답 본문 없음) |
| 400 | 요청 파라미터가 잘못되었습니다 |
| 401 | 인증 실패: 로그인하지 않은 사용자의 접근입니다 |
| 403 | 권한 실패: 본인의 리소스에만 접근할 수 있습니다 |
| 404 | 해당 리소스를 찾을 수 없습니다 |
| 500 | 서버 내부 오류가 발생했습니다 |
