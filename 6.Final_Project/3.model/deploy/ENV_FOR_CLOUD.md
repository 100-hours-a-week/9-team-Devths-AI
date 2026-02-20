# 배포 환경변수 (클라우드 팀 주입용)

AI 모델 서버 배포 시 아래 환경변수를 앱 프로세스에 주입해 주세요.

## ChromaDB 영속 경로 (필수)

| 환경변수 | 값 | 설명 |
|----------|-----|------|
| `CHROMA_PERSIST_DIR` | `/home/ubuntu/ai/chroma_db` | ChromaDB 데이터 디렉터리. 배포 서버에 이미 해당 경로에 DB가 있으면 동일 경로로 설정. |

- 로컬/미설정 시 기본값: `./chroma_db` (프로젝트 내 상대 경로)
- 배포 서버에서는 **반드시 위 값으로 설정**해 기존 Chroma DB를 사용하도록 해 주세요.

## 예시 (쉘)

```bash
export CHROMA_PERSIST_DIR=/home/ubuntu/ai/chroma_db
```

## 예시 (systemd / .env 파일)

```ini
Environment="CHROMA_PERSIST_DIR=/home/ubuntu/ai/chroma_db"
```

또는 `.env` 파일에 한 줄:

```
CHROMA_PERSIST_DIR=/home/ubuntu/ai/chroma_db
```
