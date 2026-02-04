import asyncio
import os
import time
from typing import List, Dict, Any
import boto3
from botocore.exceptions import BotoCoreError, ClientError

class CloudWatchService:
    _instance = None
    _buffer: List[Dict[str, Any]] = []
    _batch_size = 20
    _namespace = os.getenv("CW_NAMESPACE", "Devths/AI")
    _enabled = os.getenv("CW_ENABLED", "false").lower() == "true"
    _region = os.getenv("AWS_REGION", "ap-northeast-2")
    _environment = os.getenv("DEPLOYMENT_ENV", "dev") # dev, stg, prod

    _executor = None

    def __init__(self):
        if self._enabled:
            try:
                self.client = boto3.client("cloudwatch", region_name=self._region)
                import concurrent.futures
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                print(f"✅ CloudWatchService initialized (Namespace: {self._namespace}, Env: {self._environment})")
            except (BotoCoreError, ClientError) as e:
                print(f"❌ Failed to initialize CloudWatch client: {e}")
                self._enabled = False
        else:
            print("⚠️ CloudWatch monitoring is DISABLED (CW_ENABLED != true)")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CloudWatchService()
        return cls._instance

    async def put_metric(self, name: str, value: float, unit: str = "Count", dimensions: Dict[str, str] = None):
        """
        메트릭을 버퍼에 추가하고, 배치 크기가 되면 비동기(Thread Pool) 전송
        """
        if not self._enabled:
            return

        if dimensions is None:
            dimensions = {}
        
        # 기본 Dimension 추가
        dimensions["Environment"] = self._environment
        
        dims_list = [{"Name": k, "Value": v} for k, v in dimensions.items()]

        metric_data = {
            "MetricName": name,
            "Dimensions": dims_list,
            "Timestamp": time.time(),
            "Value": value,
            "Unit": unit,
            "StorageResolution": 60 # 1분 단위 (Standard Resolution)
        }

        self._buffer.append(metric_data)

        if len(self._buffer) >= self._batch_size:
            await self.flush()

    async def flush(self):
        """
        버퍼에 있는 메트릭을 강제로 전송 (비동기 처리)
        """
        if not self._enabled or not self._buffer:
            return

        metrics_to_send = self._buffer[:]
        self._buffer = []

        loop = asyncio.get_running_loop()
        if self._executor:
            try:
                # Blocking I/O를 별도 스레드에서 실행하여 이벤트 루프 차단 방지
                await loop.run_in_executor(self._executor, self._send_batch_sync, metrics_to_send)
            except Exception as e:
                print(f"❌ Failed to run metric flush in executor: {e}")

    def _send_batch_sync(self, metrics: List[Dict[str, Any]]):
        """
        실제 AWS API 호출을 수행하는 동기 함수 (스레드 풀에서 실행됨)
        """
        try:
            # CloudWatch put_metric_data는 최대 20개까지 한 번에 전송 가능
            for i in range(0, len(metrics), 20):
                batch = metrics[i:i+20]
                self.client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=batch
                )
            # print(f"🚀 Sent {len(metrics)} metrics to CloudWatch")
        except (BotoCoreError, ClientError) as e:
            print(f"❌ Failed to send metrics to CloudWatch: {e}")
            # 에러 발생 시 버퍼에 다시 넣거나 로그만 남김 (여기선 로그만)
