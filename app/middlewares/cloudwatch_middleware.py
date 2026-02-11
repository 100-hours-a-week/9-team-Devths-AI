import asyncio
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.cloudwatch_service import CloudWatchService


class CloudWatchMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.cw_service = CloudWatchService.get_instance()

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise e
        finally:
            process_time = (time.time() - start_time) * 1000  # ms 단위

            # API 경로 그룹화 (Cardinality 제어) - 중요: ID 대신 {id} 패턴 사용
            route_obj = request.scope.get("route")
            path = route_obj.path if route_obj else request.url.path

            # 메트릭 차원(Dimensions)
            dimensions = {"Method": request.method, "Path": path, "Status": str(status_code)}

            try:
                # 1. Latency (ms)
                # await 대신 create_task로 변경하여 메트릭 전송이 응답 시간을 지연시키지 않도록 함 (Fire-and-forget)
                asyncio.create_task(
                    self.cw_service.put_metric("Latency", process_time, "Milliseconds", dimensions)
                )

                # 2. Request Count
                asyncio.create_task(
                    self.cw_service.put_metric("RequestCount", 1, "Count", dimensions)
                )
            except Exception as e:
                # 메트릭 전송 실패가 API 응답에 영향을 주면 안 됨
                print(f"⚠️ Failed to queue metrics: {e}")
