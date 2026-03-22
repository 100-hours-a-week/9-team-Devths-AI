"""
OpenTelemetry 초기화 모듈

OTEL_* 환경변수로 설정 (kustomize overlay에서 주입):
  - OTEL_SERVICE_NAME
  - OTEL_EXPORTER_OTLP_ENDPOINT  (예: http://alloy.monitoring.svc.cluster.local:4318)
  - OTEL_EXPORTER_OTLP_PROTOCOL  (http/protobuf)
  - OTEL_RESOURCE_ATTRIBUTES      (deployment.environment=dev,service.namespace=devths,...)
  - OTEL_TRACES_SAMPLER           (parentbased_always_on)
  - OTEL_PROPAGATORS              (tracecontext,baggage)

OTEL_EXPORTER_OTLP_ENDPOINT 미설정 시 no-op (로컬 개발 환경 자동 비활성화).
"""

import logging
import os

logger = logging.getLogger(__name__)

_initialized = False


def setup_tracing() -> None:
    """
    OTel TracerProvider 초기화.

    FastAPI, httpx, Celery를 자동 계측합니다.
    OTEL_EXPORTER_OTLP_ENDPOINT 미설정 시 조용히 종료 (에러 없음).
    FastAPI 앱 계측은 instrument_fastapi_app()을 별도 호출해야 합니다.
    """
    global _initialized
    if _initialized:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT 미설정 — 트레이싱 비활성화")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.celery import CeleryInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON

        service_name = os.getenv("OTEL_SERVICE_NAME", "devths-ai")

        # OTEL_RESOURCE_ATTRIBUTES 파싱 (key=value,key=value 형식)
        resource_attrs = {"service.name": service_name}
        raw_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        for pair in raw_attrs.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                resource_attrs[k.strip()] = v.strip()

        resource = Resource.create(resource_attrs)
        provider = TracerProvider(resource=resource, sampler=ALWAYS_ON)

        # OTLP HTTP exporter — SDK가 /v1/traces 경로를 자동으로 추가
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        # httpx 자동 계측 (외부 API 호출 — BE, Gemini, OpenAI 등)
        HTTPXClientInstrumentor().instrument()

        # Celery 자동 계측 (Worker/Beat 프로세스에서 호출 시 태스크 span 생성)
        CeleryInstrumentor().instrument()

        _initialized = True
        logger.info(
            "OTel 트레이싱 초기화 완료 — service=%s, endpoint=%s",
            service_name,
            endpoint,
        )

    except ImportError as e:
        logger.warning("OTel 패키지 없음 — 트레이싱 비활성화: %s", e)


def instrument_fastapi_app(app) -> None:
    """
    FastAPI 앱에 OTel 계측 적용.

    setup_tracing() 호출 후, FastAPI app 생성 직후 호출해야 합니다.
    OTEL_EXPORTER_OTLP_ENDPOINT 미설정 시 no-op.
    """
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
        )
        logger.info("FastAPI OTel 계측 적용 완료")
    except ImportError as e:
        logger.warning("OTel FastAPI 계측 실패: %s", e)
