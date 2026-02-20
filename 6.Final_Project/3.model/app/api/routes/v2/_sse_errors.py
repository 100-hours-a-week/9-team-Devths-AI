"""
v2 SSE 에러 이벤트 통일 포맷

백엔드(Spring)가 SSE 스트리밍 중 에러를 인식할 수 있도록
모든 에러를 통일된 JSON 포맷으로 전송합니다.

## SSE 이벤트 타입 규약

| type           | 설명                        | 백엔드 처리              |
|----------------|-----------------------------|--------------------------|
| chunk          | 정상 텍스트 청크            | 화면에 표시              |
| summary        | 채팅방 제목 메타데이터      | 제목 업데이트            |
| session_state  | 면접 세션 상태              | 세션 상태 업데이트       |
| error          | 에러 발생                   | 에러 처리 (페이지 이동 허용 등) |
| [DONE]         | 스트림 종료                 | SSE 연결 종료            |

## 에러 이벤트 포맷

```json
{
    "type": "error",
    "error": {
        "code": "INTERNAL_ERROR",    // ErrorCode enum 값
        "status": 500,               // HTTP 상태 코드
        "message": "내부 서버 오류"   // 상세 메시지 (로깅용)
    },
    "fallback": "사용자에게 표시할 메시지"
}
```
"""

import json


def sse_error_event(
    code: str,
    status: int,
    message: str,
    fallback: str | None = None,
) -> str:
    """통일된 SSE 에러 이벤트 문자열 생성

    Args:
        code: 에러 코드 (ErrorCode enum 값, 예: "INTERNAL_ERROR", "LLM_UNAVAILABLE")
        status: HTTP 상태 코드 (예: 500, 503, 400)
        message: 상세 에러 메시지 (로깅/디버깅용)
        fallback: 사용자에게 표시할 메시지 (없으면 message 사용)

    Returns:
        SSE 포맷 문자열: "data: {...}\n\n"
    """
    error_payload = {
        "type": "error",
        "error": {
            "code": code,
            "status": status,
            "message": message,
        },
        "fallback": fallback or message,
    }
    return f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
