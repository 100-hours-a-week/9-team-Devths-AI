# 🐾 동물 감정일기 - KakaoTech Bootcamp FastAPI Project

> **바닐라 JavaScript 프론트엔드 + FastAPI 백엔드 + AI 모델 서빙**을 통합한 커뮤니티 게시판

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gemini](https://img.shields.io/badge/Google-Gemini_API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

---

## 📋 프로젝트 소개

**동물 감정일기**는 **카카오테크 부트캠프** FastAPI 프로젝트로,  
**바닐라 JavaScript** 기반 웹 프론트엔드에서 **FastAPI로 서빙되는 AI 모델**을 활용하는 커뮤니티 게시판입니다.

### 🎯 프로젝트 목표

1. ✅ **FastAPI 기반 REST API 서버** 구현
2. ✅ **바닐라 JS 웹 프론트엔드**에서 FastAPI 모델 서빙 API 사용
3. ✅ **AI 모델 연동** (이미지 분류 + 감정 분석)
4. ✅ **pytest 기반 테스트 케이스** 작성 (263개 테스트, 93.2% 통과율)
5. ✅ **Git Repository README** 프로젝트 문서화

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        사용자 (브라우저)                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│   🌐 Frontend (localhost:3000)                                       │
│   ───────────────────────────────────────────                       │
│   • Vanilla JavaScript (ES6+)                                        │
│   • HTML5 + CSS3 (SPA 구조)                                          │
│   • Fetch API로 REST 통신                                            │
└─────────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│   🔧 Backend API (port 8000)  │   │   🤖 Model API (port 8001)    │
│   ─────────────────────────── │   │   ─────────────────────────── │
│   • FastAPI REST Server       │   │   • FastAPI AI Model 서빙      │
│   • 회원가입/로그인            │   │   • 이미지 분류 (Keras CNN)     │
│   • 게시글/댓글 CRUD          │   │   • 감정 분석 (Gemini API)      │
│   • MySQL 8.0 + SQLAlchemy    │   │   • LLM 채팅 (Ollama)          │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                           │
                    └───────────┬───────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   💾 Database         │
                    │   MySQL 8.0           │
                    └───────────────────────┘
```

---

## ✨ 주요 기능

### 🌐 프론트엔드 (Vanilla JS)

| 기능 | 설명 | 연동 API |
|------|------|----------|
| **회원가입/로그인** | 이메일, 비밀번호, 닉네임 | Backend `/api/auth/*` |
| **게시글 CRUD** | 작성, 조회, 수정, 삭제 | Backend `/api/posts/*` |
| **이미지 업로드** | 게시글 이미지 첨부 + **AI 분류** | Model `/api/predict` |
| **댓글 기능** | 게시글별 댓글 + **감정 분석** | Model `/api/sentiment/gemini` |
| **좋아요/조회수** | 게시글 인터랙션 | Backend `/api/posts/{id}/like` |

### 🤖 AI 모델 서빙 (FastAPI)

| 모델 | 입력 | 출력 | 기술 스택 |
|------|------|------|-----------|
| **이미지 분류** | 이미지 파일 | 🐕 Dog / 🐈 Cat | Keras CNN |
| **감정 분석** | 텍스트 (한글/영어) | 😊 긍정 / 😞 부정 / 😐 중립 | Google Gemini 2.5 |
| **텍스트 요약** | 긴 텍스트 | 요약된 텍스트 | Gemini API |
| **LLM 채팅** | 대화 메시지 | 스트리밍 응답 | Ollama (gemma3:4b) |

---

## 🛠 기술 스택

### Frontend
| 분류 | 기술 |
|------|------|
| **Language** | Vanilla JavaScript (ES6+, 프레임워크 미사용) |
| **Markup** | HTML5, CSS3 |
| **Design** | Google Fonts (Noto Sans KR), 반응형 웹 디자인 |
| **HTTP Client** | Fetch API (async/await) |

### Backend
| 분류 | 기술 |
|------|------|
| **Framework** | FastAPI 0.104+ |
| **Language** | Python 3.10+ |
| **Database** | MySQL 8.0 |
| **ORM** | SQLAlchemy 2.0 |
| **Validation** | Pydantic |

### AI/ML
| 분류 | 기술 |
|------|------|
| **이미지 분류** | TensorFlow / Keras (CNN) |
| **감정 분석** | Google Gemini 2.5 Flash |
| **LLM** | Ollama (gemma3:4b) |

---

## 📁 프로젝트 구조

```
3.FASTAPI_PROJECT_KTB/
├── FASTAPI_Project_front/       # 🌐 프론트엔드 (Vanilla JS)
│   ├── index.html               # SPA 메인 페이지
│   ├── css/index.css            # 스타일시트
│   ├── js/
│   │   ├── api.js               # API 통신 모듈
│   │   ├── auth.js              # 인증 로직
│   │   ├── posts.js             # 게시글 로직
│   │   └── app.js               # 앱 초기화
│   └── tests/                   # pytest 테스트
│
├── FASTAPI_Project_back/        # 🔧 백엔드 API
│   ├── app/
│   │   ├── main.py              # FastAPI 앱 진입점
│   │   ├── routers/             # API 라우터
│   │   ├── controllers/         # 비즈니스 로직
│   │   ├── models/              # SQLAlchemy 모델
│   │   └── schemas.py           # Pydantic 스키마
│   └── tests/                   # pytest 테스트 (80개)
│
├── FASTAPI_Project_model/       # 🤖 AI 모델 서빙
│   ├── app/
│   │   ├── main.py              # FastAPI 앱 진입점
│   │   ├── routers/
│   │   │   ├── predict_routes.py    # 이미지 분류
│   │   │   ├── sentiment_routes.py  # 감정 분석
│   │   │   └── chat_routes.py       # LLM 채팅
│   │   └── services/
│   │       ├── model_service.py     # Keras 모델
│   │       └── gemini_service.py    # Gemini API
│   └── tests/                   # pytest 테스트 (83개)
│
├── TEST_CASES_COMPLETE.html     # 📋 통합 테스트 문서
└── README.md                    # 📖 프로젝트 문서
```

---

## 🚀 실행 방법

### 1️⃣ 저장소 클론

```bash
git clone https://github.com/yoondonggyu/KakaoTechBootcamp-FastAPI.git
cd KakaoTechBootcamp-FastAPI
```

### 2️⃣ Backend 서버 실행 (포트 8000)

```bash
cd FASTAPI_Project_back
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# DATABASE_URL 설정

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

### 3️⃣ Model 서버 실행 (포트 8001)

```bash
cd FASTAPI_Project_model
pip install -r requirements.txt

# 환경 변수 설정 (GEMINI_API_KEY)
cp .env.example .env

# 서버 실행
uvicorn app.main:app --reload --port 8001
```

### 4️⃣ Frontend 실행 (포트 3000)

```bash
cd FASTAPI_Project_front
python -m http.server 3000
# 또는
npx serve -p 3000
```

### 5️⃣ 브라우저 접속

```
http://localhost:3000
```

---

## 📚 API 명세

### Backend API (localhost:8000)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/auth/signup` | 회원가입 |
| POST | `/api/auth/login` | 로그인 |
| GET | `/api/posts` | 게시글 목록 |
| POST | `/api/posts` | 게시글 작성 |
| POST | `/api/posts/upload` | 이미지 업로드 |
| POST | `/api/posts/{id}/like` | 좋아요 토글 |

### Model API (localhost:8001)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/predict` | 이미지 분류 (Dog/Cat) |
| POST | `/api/sentiment` | 감정 분석 (영어) |
| POST | `/api/sentiment/gemini` | 감정 분석 (한글/영어) |
| POST | `/api/chat` | LLM 채팅 |
| POST | `/api/summarize` | 텍스트 요약 |

---

## 🧪 테스트

### 테스트 실행

```bash
# Backend 테스트 (80개)
cd FASTAPI_Project_back && pytest -v

# Model 테스트 (83개)
cd FASTAPI_Project_model && pytest -v

# Frontend 테스트 (100개)
cd FASTAPI_Project_front && pytest -v
```

### 테스트 결과

| 서비스 | 테스트 수 | 결과 |
|--------|----------|------|
| **Backend** | 80개 | ✅ 80 Pass (100%) |
| **Model** | 83개 | ✅ 65 Pass + ⏭️ 18 Skip |
| **Frontend** | 100개 | ✅ 100 Pass (100%) |
| **전체** | 263개 | **93.2% 통과율** |

> 📋 상세 테스트 문서: [TEST_CASES_COMPLETE.html](./TEST_CASES_COMPLETE.html)

---

## 🔗 관련 저장소

| 저장소 | 설명 | 링크 |
|--------|------|------|
| **Frontend** | Vanilla JS 웹 UI | [KakaoTechBootcamp-Frontend](https://github.com/yoondonggyu/KakaoTechBootcamp-Frontend) |
| **Backend** | FastAPI REST API | [KakaoTechBootcamp-Backend](https://github.com/yoondonggyu/KakaoTechBootcamp-Backend) |
| **Model** | AI 모델 서빙 | [KakaoTechBootcamp-Model](https://github.com/yoondonggyu/KakaoTechBootcamp-Model) |

---

## 📸 스크린샷

### 로그인 / 회원가입
- 이메일, 비밀번호 유효성 검사
- 프로필 이미지 업로드

### 게시글 목록
- 카드 형태 UI
- 좋아요, 댓글 수, 조회수 표시

### 게시글 상세
- 🐕 / 🐈 AI 이미지 분류 결과 표시
- 😊 / 😞 / 😐 댓글 감정 분석 표시

---

## � 향후 발전 방향

### 🐾 나의 반려동물과 소통하는 감정 일기장

현재의 **동물 감정일기**를 더욱 발전시켜, **반려동물과의 특별한 순간을 기록하고 소통하는 일기장 플랫폼**으로 확장할 예정입니다.

#### 🎯 발전 계획

| 단계 | 기능 | 설명 |
|------|------|------|
| **Phase 1** | 🐕 **반려동물 프로필** | 반려동물별 프로필 등록 (이름, 종, 나이, 특성) |
| **Phase 2** | 📝 **일기 작성** | 반려동물과의 하루를 기록하는 감정 일기 |
| **Phase 3** | 🤖 **AI 감정 분석** | 일기 내용을 분석하여 반려동물의 감정 상태 추적 |
| **Phase 4** | 💬 **AI 반려동물 대화** | 일기 기록을 학습한 AI가 반려동물의 입장에서 답변 |
| **Phase 5** | 📊 **감정 트렌드** | 반려동물의 감정 변화를 시각화 (그래프, 통계) |
| **Phase 6** | 🏥 **건강 관리** | 병원 기록, 예방접종, 투약 일정 관리 |
| **Phase 7** | 🎨 **추억 앨범** | 반려동물 사진 자동 분류 및 앨범 생성 |
| **Phase 8** | 👥 **커뮤니티** | 같은 품종 반려동물 보호자들과의 소통 |

#### ✨ 핵심 가치

> **"반려동물의 감정을 이해하고, 더 깊은 유대감을 형성하는 디지털 일기장"**

- 📖 **기록**: 소중한 순간을 놓치지 않고 기록
- 🧠 **이해**: AI가 반려동물의 감정과 패턴을 분석
- 💬 **소통**: 반려동물의 입장에서 생각해볼 수 있는 AI 대화
- 🤝 **공유**: 비슷한 경험을 가진 보호자들과의 정보 교류

---

## �👨‍💻 개발자

- **윤동규** (Yoon Dong-Gyu)
- GitHub: [@yoondonggyu](https://github.com/yoondonggyu)


