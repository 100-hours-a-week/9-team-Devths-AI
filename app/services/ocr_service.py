"""
OCR Service with Fallback Strategy

Primary: EasyOCR (무료, 빠름, 91.4%)
Fallback: Gemini VLM (고정확도, 97.4%)

Based on: 4.DOCS/0.제출문서/0.모델 선정/02_OCR_모델_선정.md
"""

import asyncio
import io
import logging
import os
import tempfile
from typing import Any

import httpx
import numpy as np
import pdf2image
from PIL import Image

from app.utils.langfuse_client import create_generation, trace_llm_call

logger = logging.getLogger(__name__)


class OCRService:
    """OCR Service with Fallback Strategy

    Primary: EasyOCR (무료, 빠름, 91.4%)
    Fallback: Gemini VLM (고정확도, 97.4%)

    Fallback 조건:
    - EasyOCR 실패 시
    - 추출된 텍스트가 너무 짧을 때 (50자 미만)
    - 한글 비율이 너무 낮을 때 (10% 미만)
    """

    def __init__(self, llm_service=None):
        """
        Initialize OCR Service

        Args:
            llm_service: LLMService instance for Gemini fallback
        """
        self.llm_service = llm_service
        self.reader = None
        self._init_easyocr()

    def _init_easyocr(self):
        """EasyOCR 초기화 (한국어 + 영어)"""
        try:
            from easyocr import Reader

            # GPU 사용 시도, 실패하면 CPU로 폴백
            try:
                self.reader = Reader(["ko", "en"], gpu=True)
                logger.info("OCRService: EasyOCR 초기화 완료 (GPU 모드)")
            except Exception as e:
                logger.warning(f"OCRService: EasyOCR GPU 모드 실패, CPU로 재시도: {e}")
                self.reader = Reader(["ko", "en"], gpu=False)
                logger.info("OCRService: EasyOCR 초기화 완료 (CPU 모드)")
        except ImportError:
            logger.warning("OCRService: EasyOCR 미설치 - Gemini만 사용 가능")
            self.reader = None
        except Exception as e:
            logger.error(f"OCRService: EasyOCR 초기화 실패: {e}")
            self.reader = None

    async def extract_text(
        self,
        file_url: str,
        file_type: str = "pdf",
        user_id: str | None = None,
        fallback_enabled: bool = True,
    ) -> dict[str, Any]:
        """
        OCR 텍스트 추출 (Fallback 전략 적용)

        1. EasyOCR 시도
        2. 품질 검증 (텍스트 길이, 한글 비율)
        3. 실패 또는 품질 낮음 → Gemini Fallback

        Args:
            file_url: 파일 URL 또는 S3 키
            file_type: "pdf" 또는 "image"
            user_id: 사용자 ID (Langfuse 추적용)
            fallback_enabled: Gemini 폴백 활성화 여부

        Returns:
            {
                "extracted_text": str,
                "pages": [{"page": int, "text": str}, ...],
                "ocr_engine": "easyocr" | "gemini",
                "fallback_reason": str | None  # 폴백 이유 (있으면)
            }
        """
        # EasyOCR 시도
        if self.reader is not None:
            result = await self._try_easyocr(file_url, file_type, user_id)

            if result.get("success"):
                # 품질 검증
                should_fallback, reason = self._should_fallback(result)

                if not should_fallback:
                    result["ocr_engine"] = "easyocr"
                    result["fallback_reason"] = None
                    return result

                # Fallback 필요
                logger.info(f"OCRService: EasyOCR 품질 부족 ({reason}) → Gemini Fallback")
                result["fallback_reason"] = reason
            else:
                reason = result.get("error", "EasyOCR 실패")
                logger.info(f"OCRService: EasyOCR 실패 ({reason}) → Gemini Fallback")
        else:
            reason = "EasyOCR 미초기화"
            logger.info(f"OCRService: {reason} → Gemini Fallback")

        # Gemini Fallback
        if fallback_enabled and self.llm_service:
            try:
                gemini_result = await self.llm_service.extract_text_from_file(
                    file_url=file_url,
                    file_type=file_type,
                    user_id=user_id,
                )
                gemini_result["ocr_engine"] = "gemini"
                gemini_result["fallback_reason"] = reason
                gemini_result["success"] = True
                return gemini_result
            except Exception as e:
                logger.error(f"OCRService: Gemini Fallback 실패: {e}")
                return {
                    "success": False,
                    "error": f"모든 OCR 엔진 실패: EasyOCR({reason}), Gemini({e})",
                    "extracted_text": "",
                    "pages": [],
                    "ocr_engine": None,
                }

        # Fallback 비활성화 또는 llm_service 없음
        return {
            "success": False,
            "error": f"EasyOCR 실패: {reason}",
            "extracted_text": "",
            "pages": [],
            "ocr_engine": None,
        }

    async def _try_easyocr(
        self,
        file_url: str,
        file_type: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """EasyOCR로 텍스트 추출"""
        try:
            # 파일 단위 trace 생성
            ocr_trace = trace_llm_call(
                name="easyocr_extract_text",
                user_id=user_id,
                metadata={
                    "type": "ocr",
                    "file_type": file_type,
                    "file_url_prefix": file_url[:100] if file_url else "",
                    "ocr_engine": "easyocr",
                },
            )

            # 파일 다운로드
            logger.info(f"[EasyOCR] 파일 다운로드 중: {file_url[:80] if file_url else ''}...")
            file_bytes = await self._download_file(file_url)

            pages = []
            full_text = ""

            if file_type == "pdf" or "pdf" in file_type.lower():
                # PDF → 이미지 변환
                logger.info("[EasyOCR] PDF → 이미지 변환 중...")
                images = self._pdf_to_images(file_bytes)

                for page_num, image in enumerate(images, start=1):
                    logger.info(f"[EasyOCR] 페이지 {page_num}/{len(images)} 처리 중...")
                    page_text = await self._ocr_image(image)
                    pages.append({"page": page_num, "text": page_text})
                    full_text += f"\n\n[Page {page_num}]\n{page_text}"

                    # Langfuse generation 기록
                    if ocr_trace is not None:
                        create_generation(
                            trace=ocr_trace,
                            name=f"easyocr_page_{page_num}",
                            model="easyocr",
                            input_text=f"file_type=pdf page={page_num}",
                            output_text=page_text[:4000],
                            metadata={
                                "type": "ocr",
                                "page": page_num,
                                "total_pages": len(images),
                                "ocr_engine": "easyocr",
                            },
                        )
            else:
                # 단일 이미지
                logger.info("[EasyOCR] 이미지 처리 중...")
                image = Image.open(io.BytesIO(file_bytes))
                text = await self._ocr_image(image)
                pages.append({"page": 1, "text": text})
                full_text = text

                if ocr_trace is not None:
                    create_generation(
                        trace=ocr_trace,
                        name="easyocr_image",
                        model="easyocr",
                        input_text="file_type=image",
                        output_text=text[:4000],
                        metadata={
                            "type": "ocr",
                            "pages": 1,
                            "ocr_engine": "easyocr",
                        },
                    )

            logger.info(f"[EasyOCR] 완료: {len(full_text)}자 추출, {len(pages)}페이지")

            return {
                "success": True,
                "extracted_text": full_text.strip(),
                "pages": pages,
            }

        except Exception as e:
            logger.error(f"[EasyOCR] 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "pages": [],
            }

    async def _ocr_image(self, image: Image.Image) -> str:
        """EasyOCR로 이미지에서 텍스트 추출"""
        # PIL Image → numpy array
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_array = np.array(image)

        # EasyOCR 실행 (비동기 처리 - blocking 호출을 executor에서 실행)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.reader.readtext(img_array, detail=0, paragraph=True),
        )

        # 결과 합치기 (paragraph=True로 이미 문단 단위로 반환됨)
        return "\n".join(result) if isinstance(result, list) else str(result)

    def _should_fallback(self, result: dict) -> tuple[bool, str | None]:
        """
        Fallback 필요 여부 판단

        Returns:
            (should_fallback: bool, reason: str | None)
        """
        if not result.get("success"):
            return True, "추출 실패"

        text = result.get("extracted_text", "")

        # 1. 추출된 텍스트가 너무 짧으면 Fallback
        if len(text) < 50:
            return True, f"텍스트 길이 부족 ({len(text)}자)"

        # 2. 한글 비율이 너무 낮으면 Fallback (한국어 문서인데 인식 실패 가능성)
        korean_chars = sum(1 for c in text if "가" <= c <= "힣")
        if len(text) > 100 and korean_chars / len(text) < 0.1:
            return (
                True,
                f"한글 비율 낮음 ({korean_chars}/{len(text)} = {korean_chars/len(text):.1%})",
            )

        return False, None

    async def _download_file(self, file_url: str) -> bytes:
        """파일 다운로드 (URL 또는 S3 키)

        Supports:
        - data: URLs (base64 encoded)
        - http(s):// URLs (직접 다운로드)
        - S3 keys (e.g., "uploads/2026/01/xxx.png") - S3_BASE_URL로 변환
        """
        import base64

        # data: URL 처리
        if file_url.startswith("data:"):
            try:
                header, encoded = file_url.split(",", 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"data URL 디코딩 실패: {e}")
                raise ValueError("Invalid data URL format") from e

        # HTTP(S) URL 직접 다운로드
        if file_url.startswith("http://") or file_url.startswith("https://"):
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url, timeout=60.0)
                response.raise_for_status()
                return response.content

        # S3 키 → 전체 URL 변환
        s3_base_url = os.getenv("S3_BASE_URL", "").rstrip("/")
        if s3_base_url:
            full_url = f"{s3_base_url}/{file_url.lstrip('/')}"
            logger.info(f"S3 키 → URL 변환: {file_url[:50]}...")
            async with httpx.AsyncClient() as client:
                response = await client.get(full_url, timeout=60.0)
                response.raise_for_status()
                return response.content

        raise ValueError(
            f"파일 다운로드 불가: S3_BASE_URL 미설정, '{file_url[:50]}...'는 유효한 URL이 아님"
        )

    def _pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
        """PDF를 이미지로 변환"""
        Image.MAX_IMAGE_PIXELS = None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            images = pdf2image.convert_from_path(tmp_path, dpi=dpi)
            logger.info(f"PDF → {len(images)}개 이미지 변환 완료")
            return images
        finally:
            os.unlink(tmp_path)
