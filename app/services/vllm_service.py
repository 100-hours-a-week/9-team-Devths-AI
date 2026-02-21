"""
vLLM Service for Llama-3-Korean-Bllossom-8B

Provides chat functionality and OCR using vLLM API server.
"""

import io
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pdf2image
from PIL import Image

from app.utils.langfuse_client import create_generation, trace_llm_call

logger = logging.getLogger(__name__)


class VLLMService:
    """vLLM Service for chatbot using GCP GPU server"""

    def __init__(
        self, base_url: str | None = None, ocr_only: bool = False, model_name: str | None = None
    ):
        """
        Initialize vLLM Service

        Args:
            base_url: GCP vLLM server URL (default: from GCP_VLLM_BASE_URL env)
            ocr_only: If True, only OCR functionality is available
            model_name: Model name (default: from VLLM_MODEL_NAME env)
        """
        self.ocr_only = ocr_only

        if ocr_only:
            logger.info("🔧 vLLM Service initialized in OCR-only mode (EasyOCR)")
            logger.info("💰 가성비 모드: vLLM 서버 없이 EasyOCR만 사용")
            self.base_url = None
            self.model_name = "easyocr"
            return

        # GCP vLLM 서버 URL 읽기
        gcp_url = os.getenv("GCP_VLLM_BASE_URL")
        self.base_url = base_url or gcp_url

        if not self.base_url:
            logger.warning("⚠️ GCP_VLLM_BASE_URL이 설정되지 않았습니다. OCR 전용 모드로 전환합니다.")
            self.ocr_only = True
            self.model_name = "easyocr"
            return

        # vLLM 모델명 읽기
        self.model_name = model_name or os.getenv(
            "VLLM_MODEL_NAME", "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        )

        # 로깅
        logger.info("🌐 vLLM Service initialized with GCP GPU server")
        logger.info(f"📍 GCP URL: {self.base_url}")
        logger.info(f"📦 Model: {self.model_name}")

    def get_server_info(self) -> dict[str, Any]:
        """Get GCP vLLM server information"""
        return {
            "server_url": self.base_url,
            "model_name": self.model_name,
            "ocr_only": self.ocr_only,
        }

    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from vLLM

        Args:
            user_message: User's message
            context: RAG context from VectorDB (optional)
            history: Chat history [{"role": "user", "content": "..."}, ...]
            system_prompt: System instructions (optional)

        Yields:
            Response chunks
        """
        try:
            # Build messages
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add context if provided
            if context:
                messages.append(
                    {
                        "role": "system",
                        "content": f"관련 정보:\n{context}\n\n위 관련 정보를 참고하여 질문에 답변해주세요.",
                    }
                )

            # Add history
            if history:
                for msg in history:
                    # Handle both dict and Pydantic model
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                    else:
                        # Pydantic model (ChatMessage)
                        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                        content = msg.content

                    messages.append({"role": role, "content": content})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Prepare request
            payload = {
                "model": self.model_name,  # GCP에 떠 있는 모델명
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,  # 1024 → 2048 (응답 끊김 방지)
                "stream": True,
            }

            # Make streaming request
            # URL 검증 및 정규화
            api_url = f"{self.base_url}/v1/chat/completions"

            # URL이 유효한지 확인
            if not api_url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Invalid URL format: {api_url}. URL must start with http:// or https://"
                )

            # Colab ngrok URL은 HTTPS일 수 있으므로 verify=False 허용 (개발 환경)
            verify_ssl = os.getenv("VLLM_VERIFY_SSL", "true").lower() == "true"

            logger.debug(f"vLLM API 요청 URL: {api_url}")

            async with httpx.AsyncClient(timeout=60.0, verify=verify_ssl) as client:
                try:
                    async with client.stream(
                        "POST", api_url, json=payload, headers={"Content-Type": "application/json"}
                    ) as response:
                        # 에러 처리 먼저
                        if response.status_code != 200:
                            # 스트림 읽기 전에 에러 체크
                            try:
                                error_data = b""
                                async for chunk in response.aiter_bytes():
                                    error_data += chunk
                                error_msg = error_data.decode() if error_data else "Unknown error"
                            except Exception:
                                error_msg = f"HTTP {response.status_code}"

                            logger.error(f"vLLM API error: {response.status_code}")
                            logger.error(f"Error details: {error_msg}")
                            raise Exception(f"vLLM 서버 오류 ({response.status_code}): {error_msg}")

                        # 스트리밍 응답 처리
                        logger.info("vLLM 스트리밍 응답 수신 시작")
                        chunk_count = 0
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue

                            # SSE 형식: "data: {...}" 또는 "data: [DONE]"
                            if line.startswith("data: "):
                                data_str = line[6:].strip()  # Remove "data: " prefix

                                if data_str == "[DONE]":
                                    logger.info(f"vLLM 스트리밍 완료 (총 {chunk_count}개 청크)")
                                    break

                                try:
                                    import json

                                    data = json.loads(data_str)

                                    # Extract content from choices
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            chunk_count += 1
                                            if chunk_count <= 3:  # 처음 3개만 로깅
                                                logger.info(
                                                    f"vLLM 청크 {chunk_count}: {content[:50]}..."
                                                )
                                            yield content
                                except json.JSONDecodeError:
                                    logger.debug(f"JSON 파싱 스킵: {data_str[:50]}")
                                    continue
                                except Exception as e:
                                    logger.warning(
                                        f"Error parsing vLLM response chunk: {e}, data: {data_str[:100]}"
                                    )
                                    continue
                            elif line.strip():  # 빈 라인이 아닌 경우 로깅
                                logger.debug(f"vLLM 응답 라인 (data: 없음): {line[:100]}")

                        if chunk_count == 0:
                            logger.warning(
                                "vLLM 응답에서 청크를 받지 못했습니다. 응답 형식을 확인하세요."
                            )

                except httpx.ConnectError as e:
                    logger.error(f"vLLM 서버 연결 실패: {self.base_url}")
                    logger.error(f"연결 오류: {str(e)}")
                    try:
                        from app.core.monitoring import EXTERNAL_API_ERRORS
                        EXTERNAL_API_ERRORS.labels(provider="vllm", error_type="ConnectError").inc()
                    except Exception:
                        pass
                    raise Exception(
                        f"vLLM 서버에 연결할 수 없습니다. URL을 확인하세요: {self.base_url}"
                    ) from e
                except httpx.TimeoutException as e:
                    logger.error(f"vLLM 서버 응답 시간 초과: {self.base_url}")
                    try:
                        from app.core.monitoring import EXTERNAL_API_ERRORS
                        EXTERNAL_API_ERRORS.labels(provider="vllm", error_type="TimeoutException").inc()
                    except Exception:
                        pass
                    raise Exception("vLLM 서버 응답 시간이 초과되었습니다.") from e
                except httpx.HTTPStatusError as e:
                    logger.error(f"vLLM HTTP 오류: {e.response.status_code} - {e.response.text}")
                    try:
                        from app.core.monitoring import EXTERNAL_API_ERRORS
                        EXTERNAL_API_ERRORS.labels(provider="vllm", error_type=f"HTTP_{e.response.status_code}").inc()
                    except Exception:
                        pass
                    raise Exception(f"vLLM 서버 오류: {e.response.status_code}") from e

        except Exception as e:
            logger.error(f"Error generating vLLM response: {e}")
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def extract_text_from_file(
        self,
        file_url: str,
        file_type: str = "pdf",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Extract text from image or PDF file using EasyOCR

        Args:
            file_url: URL of the file (http(s):// or data:)
            file_type: "pdf" or "image"
            user_id: User ID for Langfuse tracking

        Returns:
            Dict with extracted_text and pages list
        """
        try:
            import numpy as np
            from easyocr import Reader

            # EasyOCR Reader 초기화 (한국어 + 영어)
            reader = Reader(["ko", "en"], gpu=True)

            # 파일 단위 trace 생성 (페이지/이미지 generation을 여기에 연결)
            ocr_trace = trace_llm_call(
                name="easyocr_extract_text",
                user_id=user_id,
                metadata={
                    "model": self.model_name,
                    "type": "ocr",
                    "file_type": file_type,
                    "file_url_prefix": file_url[:100],
                    "ocr_engine": "easyocr",
                },
            )

            # Download file
            logger.info(f"[EasyOCR] Downloading file from {file_url[:100]}...")
            file_bytes = await self._download_file(file_url)

            if file_type == "pdf":
                # Convert PDF to images
                logger.info("[EasyOCR] Converting PDF to images...")
                images = self._pdf_to_images(file_bytes)

                # Extract text from each page using EasyOCR
                pages = []
                full_text = ""

                for page_num, image in enumerate(images, start=1):
                    logger.info(f"[EasyOCR] Extracting text from page {page_num}/{len(images)}...")
                    # PIL Image를 numpy array로 변환
                    img_array = np.array(image)
                    results = reader.readtext(img_array)
                    # EasyOCR 결과에서 텍스트만 추출
                    page_text = "\n".join([text for (bbox, text, conf) in results])
                    pages.append({"page": page_num, "text": page_text})
                    full_text += f"\n\n[Page {page_num}]\n{page_text}"

                    # 페이지 단위 generation 기록 (trace가 있을 때만)
                    if ocr_trace is not None:
                        create_generation(
                            trace=ocr_trace,
                            name=f"easyocr_page_{page_num}",
                            model=self.model_name,
                            input_text=f"file_type=pdf page={page_num} file_url_prefix={file_url[:100]}",
                            output_text=page_text[:4000],
                            metadata={
                                "type": "ocr",
                                "file_type": "pdf",
                                "page": page_num,
                                "total_pages": len(images),
                                "ocr_engine": "easyocr",
                            },
                        )

                result = {"extracted_text": full_text.strip(), "pages": pages}

                # 파일 요약 generation (전체 텍스트는 너무 길 수 있어 4,000자까지만 저장)
                if ocr_trace is not None:
                    create_generation(
                        trace=ocr_trace,
                        name="easyocr_pdf_summary",
                        model=self.model_name,
                        input_text=f"file_type=pdf file_url_prefix={file_url[:100]}",
                        output_text=result["extracted_text"][:4000],
                        metadata={
                            "type": "ocr",
                            "file_type": "pdf",
                            "pages": len(pages),
                            "ocr_engine": "easyocr",
                        },
                    )

                logger.info(
                    f"[EasyOCR] Extracted {len(full_text)} characters from {len(pages)} pages"
                )
                return result
            else:
                # Single image
                logger.info("[EasyOCR] Extracting text from image...")
                image = Image.open(io.BytesIO(file_bytes))
                img_array = np.array(image)
                results = reader.readtext(img_array)
                text = "\n".join([text for (bbox, text, conf) in results])

                result = {"extracted_text": text, "pages": [{"page": 1, "text": text}]}

                if ocr_trace is not None:
                    create_generation(
                        trace=ocr_trace,
                        name="easyocr_image",
                        model=self.model_name,
                        input_text=f"file_type=image file_url_prefix={file_url[:100]}",
                        output_text=text[:4000],
                        metadata={
                            "type": "ocr",
                            "file_type": "image",
                            "pages": 1,
                            "ocr_engine": "easyocr",
                        },
                    )

                logger.info(f"[EasyOCR] Extracted {len(text)} characters from image")
                return result

        except Exception as e:
            logger.error(f"[EasyOCR] Error extracting text from file: {e}")
            raise

    async def _download_file(self, file_url: str) -> bytes:
        """Download file from URL or S3 key

        Supports:
        - data: URLs (base64 encoded)
        - http(s):// URLs (direct download)
        - S3 keys (e.g., "uploads/2026/01/xxx.png") - converted to URL using S3_BASE_URL
        """
        import os

        # Handle data: URL
        if file_url.startswith("data:"):
            try:
                import base64

                header, encoded = file_url.split(",", 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"Failed to decode data URL: {e}")
                raise ValueError("Invalid data URL format") from e

        # Handle HTTP(S) URL (direct download)
        if file_url.startswith("http://") or file_url.startswith("https://"):
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url, timeout=30.0)
                response.raise_for_status()
                return response.content

        # Handle S3 key (convert to full URL)
        # e.g., "uploads/2026/01/xxx.png" → "https://bucket.s3.amazonaws.com/uploads/2026/01/xxx.png"
        s3_base_url = os.getenv("S3_BASE_URL", "").rstrip("/")
        if s3_base_url:
            full_url = f"{s3_base_url}/{file_url.lstrip('/')}"
            logger.info(f"Converting S3 key to URL: {file_url[:50]}... → {full_url[:80]}...")
            async with httpx.AsyncClient() as client:
                response = await client.get(full_url, timeout=30.0)
                response.raise_for_status()
                return response.content

        # S3_BASE_URL not configured and not a valid URL
        raise ValueError(
            f"Cannot download file: S3_BASE_URL not configured and '{file_url[:50]}...' is not a valid URL"
        )

    def _pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
        """Convert PDF to images"""
        from PIL import Image as PILImage

        PILImage.MAX_IMAGE_PIXELS = None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            images = pdf2image.convert_from_path(tmp_path, dpi=dpi)
            logger.info(f"[vLLM OCR] Converted PDF to {len(images)} images")
            return images
        finally:
            os.unlink(tmp_path)
