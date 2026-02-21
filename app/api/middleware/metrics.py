import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from app.core.monitoring import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI의 모든 HTTP 요청을 가로채서
    요청 횟수, 처리 중인 요청 수, 전체 소요 시간(Latency)을 기록합니다.
    """

    async def dispatch(self, request: Request, call_next):
        method = request.method
        
        # FastAPI Router 패턴 매칭으로 Path 추출 (파라미터 분산을 막기 위함)
        # 예: /api/v2/chat/123 -> /api/v2/chat/{id}
        path, is_handled_path = self.get_path(request)
        if not is_handled_path:
            # 존재하지 않는 경로 거나 모니터링이 필요 없는 경로는 원본 그대로 둡니다.
            pass

        # 처리 중인 요청 수(Gauge) 증가
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path).inc()
        
        start_time = time.perf_counter()
        status_code = 500  # 예외 발생 시 기본값

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            # 처리 중인 요청 수 감소
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path).dec()
            
            # 소요 시간 측정
            process_time = time.perf_counter() - start_time
            
            # 메트릭 기록 (Counter & Histogram)
            # 상태 코드는 문자열 변환
            HTTP_REQUESTS_TOTAL.labels(
                method=method, path=path, status=str(status_code)
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=method, path=path, status=str(status_code)
            ).observe(process_time)

    def get_path(self, request: Request) -> tuple[str, bool]:
        """
        요청 경로를 템플릿 형태로 반환하여 Prometheus 라벨(Cardinality) 폭발을 방지합니다.
        """
        for route in request.app.routes:
            match, child_scope = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True
        
        return request.url.path, False
