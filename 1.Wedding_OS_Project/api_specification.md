# Wedding OS API Specification

## 1. Auth (인증)

### 1.1 로그인
| Method | Path |
| :--- | :--- |
| POST | /api/auth/login |

**Request Body (JSON)**
```json
{
  "email": "user@example.com",
  "password": "Test@1234"
}
```
**Description**
사용자가 로그인 폼에서 이메일과 비밀번호를 입력하면, 서버는 입력값 검증 후 로그인 절차를 수행합니다.

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "login_success", "data": {"user_id": 1, "nickname": "estar", "profile_image_url": "..."}}` | 로그인 성공 |
| 400 | `{"message": "invalid_credentials", "data": null}` | 아이디 또는 비밀번호를 확인 해주세요 |
| 422 | `{"message": "invalid_email_format", "data": null}` | 올바른 이메일 주소 형식을 입력해주세요 |
| 500 | `{"message": "internal_server_error", "data": null}` | 서버 오류 |

### 1.2 회원가입
| Method | Path |
| :--- | :--- |
| POST | /api/auth/signup |

**Request Body (JSON)**
```json
{
  "email": "user@example.com",
  "password": "Test@1234",
  "password_check": "Test@1234",
  "nickname": "estar",
  "profile_image_url": "https://..."
}
```
**Description**
새로운 사용자를 등록합니다.

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 201 | `{"message": "register_success", "data": {"user_id": 1}}` | 회원가입 성공 |
| 409 | `{"message": "duplicate_email", "data": null}` | 이미 존재하는 이메일입니다 |
| 409 | `{"message": "duplicate_nickname", "data": null}` | 이미 존재하는 닉네임입니다 |
| 422 | `{"message": "invalid_password_format", "data": null}` | 비밀번호 형식이 올바르지 않습니다 |

---

## 2. Users (사용자)

### 2.1 프로필 이미지 업로드
| Method | Path |
| :--- | :--- |
| POST | /api/users/profile/upload |

**Request Body (Multipart/Form-Data)**
- `file`: 이미지 파일

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "upload_success", "data": "https://..."}` | 업로드 성공 |

### 2.2 프로필 수정
| Method | Path |
| :--- | :--- |
| PATCH | /api/users/profile |

**Request Body (JSON)**
```json
{
  "nickname": "new_nickname",
  "profile_image_url": "https://..."
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "update_profile_success", "data": {...}}` | 수정 성공 |

### 2.3 비밀번호 변경
| Method | Path |
| :--- | :--- |
| PUT | /api/users/password |

**Request Body (JSON)**
```json
{
  "old_password": "Old@1234",
  "password": "New@1234",
  "password_check": "New@1234"
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "update_password_success", "data": null}` | 변경 성공 |
| 400 | `{"message": "invalid_credentials", "data": null}` | 기존 비밀번호 불일치 |

---

## 3. Posts (게시판)

### 3.1 게시글 목록 조회
| Method | Path |
| :--- | :--- |
| GET | /api/posts |

**Query Parameters**
- `page`: 페이지 번호 (기본 1)
- `limit`: 페이지 당 개수 (기본 10)
- `board_type`: 게시판 타입 (couple, etc.)

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "get_posts_success", "data": {"posts": [...], "total": 100}}` | 조회 성공 |

### 3.2 게시글 작성
| Method | Path |
| :--- | :--- |
| POST | /api/posts |

**Request Body (JSON)**
```json
{
  "title": "게시글 제목",
  "content": "내용",
  "image_url": "https://...",
  "board_type": "couple"
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 201 | `{"message": "create_post_success", "data": {"post_id": 1, ...}}` | 작성 성공 |

### 3.3 게시글 상세 조회
| Method | Path |
| :--- | :--- |
| GET | /api/posts/{post_id} |

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "get_post_success", "data": {...}}` | 조회 성공 |
| 404 | `{"message": "post_not_found", "data": null}` | 게시글 없음 |

---

## 4. Calendar (캘린더)

### 4.1 예식일 설정
| Method | Path |
| :--- | :--- |
| POST | /api/calendar/wedding-date |

**Request Body (JSON)**
```json
{
  "wedding_date": "2025-12-25"
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "wedding_date_set", "data": {"wedding_date": "2025-12-25"}}` | 설정 성공 |

### 4.2 타임라인 자동 생성
| Method | Path |
| :--- | :--- |
| POST | /api/calendar/timeline/generate |

**Request Body (JSON)**
```json
{
  "wedding_date": "2025-12-25",
  "user_preferences": {}
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "timeline_generated", "data": {"events_count": 10}}` | 생성 성공 |

### 4.3 일정 생성
| Method | Path |
| :--- | :--- |
| POST | /api/calendar/events |

**Request Body (JSON)**
```json
{
  "title": "드레스 투어",
  "start_date": "2025-06-01",
  "start_time": "14:00",
  "category": "fitting",
  "priority": "high",
  "assignee": "bride"
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "event_created", "data": {...}}` | 생성 성공 |

### 4.4 일정 목록 조회
| Method | Path |
| :--- | :--- |
| GET | /api/calendar/events |

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "events_retrieved", "data": {"events": [...]}}` | 조회 성공 |

---

## 5. Budget (예산)

### 5.1 예산 항목 생성
| Method | Path |
| :--- | :--- |
| POST | /api/budget/items |

**Request Body (JSON)**
```json
{
  "item_name": "웨딩홀 대관료",
  "category": "hall",
  "estimated_budget": 3000000,
  "payer": "both"
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "budget_item_created", "data": {...}}` | 생성 성공 |

### 5.2 예산 요약 조회
| Method | Path |
| :--- | :--- |
| GET | /api/budget/summary |

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "budget_summary_retrieved", "data": {"total_budget": 50000000, "total_estimated": 45000000, ...}}` | 조회 성공 |

### 5.3 영수증 OCR 처리
| Method | Path |
| :--- | :--- |
| POST | /api/budget/process-receipt |

**Request Body (Multipart/Form-Data)**
- `file`: 영수증 이미지

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "receipt_processed", "data": {"item_name": "...", "amount": 100000}}` | 처리 성공 |

---

## 6. Chat (AI 플래너)

### 6.1 대화 (스트리밍)
| Method | Path |
| :--- | :--- |
| POST | /api/chat |

**Request Body (JSON)**
```json
{
  "message": "스드메 예산 얼마나 잡아야 해?",
  "include_context": true
}
```

**Responses**
| Status | Body (Stream) | Description |
| :--- | :--- | :--- |
| 200 | NDJSON Stream | 스트리밍 응답 |

---

## 7. Voice (음성 비서)

### 7.1 음성 처리
| Method | Path |
| :--- | :--- |
| POST | /api/voice/process |

**Request Body (JSON)**
```json
{
  "audio_data": "base64_encoded_audio...",
  "user_id": 1
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "voice_processed", "data": {"text": "...", "intent": "..."}}` | 처리 성공 |

---

## 8. Vendor (업체 추천)

### 8.1 결혼식 프로필 생성
| Method | Path |
| :--- | :--- |
| POST | /api/wedding-profiles |

**Request Body (JSON)**
```json
{
  "wedding_date": "2025-10-10",
  "guest_count_category": "MEDIUM",
  "total_budget": 30000000,
  "location_city": "Seoul",
  "location_district": "Gangnam"
}
```

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "profile_created", "data": {...}}` | 생성 성공 |

### 8.2 업체 추천
| Method | Path |
| :--- | :--- |
| GET | /api/vendors/recommend |

**Query Parameters**
- `wedding_profile_id`: 프로필 ID
- `vendor_type`: 업체 종류 (VENUE_OUTDOOR, etc.)

**Responses**
| Status | Body (JSON) | Description |
| :--- | :--- | :--- |
| 200 | `{"message": "vendors_recommended", "data": [...]}` | 추천 성공 |
