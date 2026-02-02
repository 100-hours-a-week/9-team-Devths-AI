"""
OCR Service with Fallback Strategy

Primary: Naver CLOVA OCR (API, 고품질)
Fallback: Gemini VLM (고정확도)

EasyOCR: GPU 인스턴스 없음으로 주석 처리 (도입 시 재활성화)

Based on: 4.DOCS/0.제출문서/0.모델 선정/02_OCR_모델_선정.md
CLOVA: https://www.ncloud.com/product/aiService/ocr
"""

import base64
import io
import logging
import os
import tempfile
import time
import uuid
from typing import Any

import httpx
import pdf2image
from PIL import Image

from app.utils.langfuse_client import create_generation, trace_llm_call

logger = logging.getLogger(__name__)

# CLOVA OCR 설정 (환경변수)
CLOVA_OCR_API_URL = os.getenv("CLOVA_OCR_API_URL", "").strip()
CLOVA_OCR_SECRET_KEY = os.getenv("CLOVA_OCR_SECRET_KEY", "").strip()
CLOVA_AVAILABLE = bool(CLOVA_OCR_API_URL and CLOVA_OCR_SECRET_KEY)


class OCRService:
    """OCR Service with Fallback Strategy

    Primary: Naver CLOVA OCR (API)
    Fallback: Gemini VLM (고정확도)

    Fallback 조건:
    - CLOVA 미설정 또는 실패 시
    - 추출된 텍스트 품질 검증 실패 시 (선택)
    """

    def __init__(self, llm_service=None):
        """
        Initialize OCR Service

        Args:
            llm_service: LLMService instance for Gemini fallback
        """
        self.llm_service = llm_service
        # EasyOCR: GPU 인스턴스 없음으로 사용 안 함 (주석 처리)
        # self.reader = None
        # self._init_easyocr()

    # ---------- EasyOCR (GPU 인스턴스 도입 후 재활성화 예정) ----------
    # def _init_easyocr(self):
    #     """EasyOCR 초기화 (한국어 + 영어)"""
    #     try:
    #         from easyocr import Reader
    #         try:
    #             self.reader = Reader(["ko", "en"], gpu=True)
    #             logger.info("OCRService: EasyOCR 초기화 완료 (GPU 모드)")
    #         except Exception as e:
    #             logger.warning(f"OCRService: EasyOCR GPU 모드 실패, CPU로 재시도: {e}")
    #             self.reader = Reader(["ko", "en"], gpu=False)
    #             logger.info("OCRService: EasyOCR 초기화 완료 (CPU 모드)")
    #     except ImportError:
    #         logger.warning("OCRService: EasyOCR 미설치 - Gemini만 사용 가능")
    #         self.reader = None
    #     except Exception as e:
    #         logger.error(f"OCRService: EasyOCR 초기화 실패: {e}")
    #         self.reader = None

    async def extract_text(
        self,
        file_url: str,
        file_type: str = "pdf",
        user_id: str | None = None,
        fallback_enabled: bool = True,
    ) -> dict[str, Any]:
        """
        OCR 텍스트 추출 (Fallback 전략 적용)

        1. CLOVA OCR 시도 (설정 시)
        2. 품질 검증 (텍스트 길이, 한글 비율)
        3. 실패 또는 품질 낮음 → Gemini Fallback

        Returns:
            {
                "extracted_text": str,
                "pages": [{"page": int, "text": str}, ...],
                "ocr_engine": "clova" | "gemini",
                "fallback_reason": str | None
            }
        """
        reason = "CLOVA 미설정"
        result = None

        if CLOVA_AVAILABLE:
            result = await self._try_clova_ocr(file_url, file_type, user_id)
            if result.get("success"):
                should_fallback, reason = self._should_fallback(result)
                if not should_fallback:
                    result["ocr_engine"] = "clova"
                    result["fallback_reason"] = None
                    return result
                logger.info(f"OCRService: CLOVA 품질 부족 ({reason}) → Gemini Fallback")
                result["fallback_reason"] = reason
            else:
                reason = result.get("error", "CLOVA 실패")
                logger.info(f"OCRService: CLOVA 실패 ({reason}) → Gemini Fallback")
        else:
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
                    "error": f"모든 OCR 엔진 실패: CLOVA({reason}), Gemini({e})",
                    "extracted_text": "",
                    "pages": [],
                    "ocr_engine": None,
                }

        return {
            "success": False,
            "error": f"CLOVA 실패: {reason}",
            "extracted_text": "",
            "pages": [],
            "ocr_engine": None,
        }

    async def _try_clova_ocr(
        self,
        file_url: str,
        file_type: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Naver CLOVA OCR로 텍스트 추출 (General API)"""
        if not CLOVA_AVAILABLE:
            return {
                "success": False,
                "error": "CLOVA OCR API 키가 설정되지 않았습니다.",
                "extracted_text": "",
                "pages": [],
            }
        try:
            ocr_trace = trace_llm_call(
                name="clova_ocr_extract_text",
                user_id=user_id,
                metadata={
                    "type": "ocr",
                    "file_type": file_type,
                    "file_url_prefix": file_url[:100] if file_url else "",
                    "ocr_engine": "clova",
                },
            )
            logger.info(f"[CLOVA OCR] 파일 다운로드 중: {file_url[:80] if file_url else ''}...")
            file_bytes = await self._download_file(file_url)
            pages = []
            full_text = ""

            if file_type == "pdf" or "pdf" in file_type.lower():
                logger.info("[CLOVA OCR] PDF → 이미지 변환 중...")
                images = self._pdf_to_images(file_bytes)
                for page_num, image in enumerate(images, start=1):
                    logger.info(f"[CLOVA OCR] 페이지 {page_num}/{len(images)} 처리 중...")
                    page_bytes, fmt = self._image_to_bytes(image, f"page_{page_num}")
                    page_text = await self._clova_ocr_image_bytes(page_bytes, fmt)
                    pages.append({"page": page_num, "text": page_text})
                    full_text += f"\n\n[Page {page_num}]\n{page_text}"
                    if ocr_trace is not None:
                        create_generation(
                            trace=ocr_trace,
                            name=f"clova_ocr_page_{page_num}",
                            model="clova_ocr",
                            input_text=f"file_type=pdf page={page_num}",
                            output_text=page_text[:4000],
                            metadata={
                                "type": "ocr",
                                "page": page_num,
                                "total_pages": len(images),
                                "ocr_engine": "clova",
                            },
                        )
            else:
                logger.info("[CLOVA OCR] 이미지 처리 중...")
                image = Image.open(io.BytesIO(file_bytes))
                name = "image"
                if file_url and "/" in file_url:
                    name = file_url.split("/")[-1][:50] or "image"
                page_bytes, fmt = self._image_to_bytes(image, name)
                text = await self._clova_ocr_image_bytes(page_bytes, fmt)
                pages.append({"page": 1, "text": text})
                full_text = text
                if ocr_trace is not None:
                    create_generation(
                        trace=ocr_trace,
                        name="clova_ocr_image",
                        model="clova_ocr",
                        input_text="file_type=image",
                        output_text=text[:4000],
                        metadata={"type": "ocr", "pages": 1, "ocr_engine": "clova"},
                    )

            logger.info(f"[CLOVA OCR] 완료: {len(full_text)}자 추출, {len(pages)}페이지")
            return {"success": True, "extracted_text": full_text.strip(), "pages": pages}
        except Exception as e:
            logger.error(f"[CLOVA OCR] 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "pages": [],
            }

    def _image_to_bytes(self, image: Image.Image, name: str) -> tuple[bytes, str]:
        """PIL Image를 JPEG/PNG bytes와 format 문자열로 반환 (CLOVA: jpg | png)"""
        buf = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        fmt = "jpg"
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), fmt

    async def _clova_ocr_image_bytes(self, image_bytes: bytes, format_type: str = "jpg") -> str:
        """CLOVA OCR General API 호출 (이미지 bytes → 텍스트)"""
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        request_json = {
            "images": [
                {
                    "format": format_type,
                    "name": "image",
                    "data": image_base64,
                }
            ],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(time.time() * 1000),
        }
        headers = {
            "X-OCR-SECRET": CLOVA_OCR_SECRET_KEY,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                CLOVA_OCR_API_URL,
                headers=headers,
                json=request_json,
                timeout=30.0,
            )
        if response.status_code != 200:
            raise RuntimeError(f"CLOVA API 오류: {response.status_code} - {response.text}")
        result = response.json()
        texts = []
        for img in result.get("images", []):
            for field in img.get("fields", []):
                texts.append(field.get("inferText", ""))
        return "\n".join(texts)

    def _should_fallback(self, result: dict) -> tuple[bool, str | None]:
        """
        Fallback 필요 여부 판단
        Returns:
            (should_fallback: bool, reason: str | None)
        """
        if not result.get("success"):
            return True, "추출 실패"
        text = result.get("extracted_text", "")
        if len(text) < 50:
            return True, f"텍스트 길이 부족 ({len(text)}자)"
        korean_chars = sum(1 for c in text if "가" <= c <= "힣")
        if len(text) > 100 and korean_chars / len(text) < 0.1:
            return (
                True,
                f"한글 비율 낮음 ({korean_chars}/{len(text)} = {korean_chars/len(text):.1%})",
            )
        return False, None

    async def _download_file(self, file_url: str) -> bytes:
        """파일 다운로드 (URL 또는 S3 키)"""
        if file_url.startswith("data:"):
            try:
                header, encoded = file_url.split(",", 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"data URL 디코딩 실패: {e}")
                raise ValueError("Invalid data URL format") from e
        if file_url.startswith("http://") or file_url.startswith("https://"):
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url, timeout=60.0)
                response.raise_for_status()
                return response.content
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

    # ---------- EasyOCR (GPU 인스턴스 도입 후 재활성화 예정, 주석 처리 유지) ----------
    # 재활성화 시: __init__에서 self.reader = None 및 self._init_easyocr() 호출, 파일 상단에 import asyncio 추가
    #
    # def _init_easyocr(self):
    #     """EasyOCR 초기화 (한국어 + 영어) - __init__에서 self.reader 설정 후 호출"""
    #     try:
    #         from easyocr import Reader
    #         try:
    #             self.reader = Reader(["ko", "en"], gpu=True)
    #             logger.info("OCRService: EasyOCR 초기화 완료 (GPU 모드)")
    #         except Exception as e:
    #             logger.warning(f"OCRService: EasyOCR GPU 모드 실패, CPU로 재시도: {e}")
    #             self.reader = Reader(["ko", "en"], gpu=False)
    #             logger.info("OCRService: EasyOCR 초기화 완료 (CPU 모드)")
    #     except ImportError:
    #         logger.warning("OCRService: EasyOCR 미설치 - Gemini만 사용 가능")
    #         self.reader = None
    #     except Exception as e:
    #         logger.error(f"OCRService: EasyOCR 초기화 실패: {e}")
    #         self.reader = None
    #
    # async def _try_easyocr(
    #     self,
    #     file_url: str,
    #     file_type: str,
    #     user_id: str | None = None,
    # ) -> dict[str, Any]:
    #     """EasyOCR로 텍스트 추출"""
    #     try:
    #         ocr_trace = trace_llm_call(
    #             name="easyocr_extract_text",
    #             user_id=user_id,
    #             metadata={
    #                 "type": "ocr",
    #                 "file_type": file_type,
    #                 "file_url_prefix": file_url[:100] if file_url else "",
    #                 "ocr_engine": "easyocr",
    #             },
    #         )
    #         logger.info(f"[EasyOCR] 파일 다운로드 중: {file_url[:80] if file_url else ''}...")
    #         file_bytes = await self._download_file(file_url)
    #         pages = []
    #         full_text = ""
    #         if file_type == "pdf" or "pdf" in file_type.lower():
    #             logger.info("[EasyOCR] PDF → 이미지 변환 중...")
    #             images = self._pdf_to_images(file_bytes)
    #             for page_num, image in enumerate(images, start=1):
    #                 logger.info(f"[EasyOCR] 페이지 {page_num}/{len(images)} 처리 중...")
    #                 page_text = await self._ocr_image(image)
    #                 pages.append({"page": page_num, "text": page_text})
    #                 full_text += f"\n\n[Page {page_num}]\n{page_text}"
    #                 if ocr_trace is not None:
    #                     create_generation(
    #                         trace=ocr_trace,
    #                         name=f"easyocr_page_{page_num}",
    #                         model="easyocr",
    #                         input_text=f"file_type=pdf page={page_num}",
    #                         output_text=page_text[:4000],
    #                         metadata={
    #                             "type": "ocr",
    #                             "page": page_num,
    #                             "total_pages": len(images),
    #                             "ocr_engine": "easyocr",
    #                         },
    #                     )
    #         else:
    #             logger.info("[EasyOCR] 이미지 처리 중...")
    #             image = Image.open(io.BytesIO(file_bytes))
    #             text = await self._ocr_image(image)
    #             pages.append({"page": 1, "text": text})
    #             full_text = text
    #             if ocr_trace is not None:
    #                 create_generation(
    #                     trace=ocr_trace,
    #                     name="easyocr_image",
    #                     model="easyocr",
    #                     input_text="file_type=image",
    #                     output_text=text[:4000],
    #                     metadata={"type": "ocr", "pages": 1, "ocr_engine": "easyocr"},
    #                 )
    #         logger.info(f"[EasyOCR] 완료: {len(full_text)}자 추출, {len(pages)}페이지")
    #         return {"success": True, "extracted_text": full_text.strip(), "pages": pages}
    #     except Exception as e:
    #         logger.error(f"[EasyOCR] 처리 실패: {e}")
    #         return {
    #             "success": False,
    #             "error": str(e),
    #             "extracted_text": "",
    #             "pages": [],
    #         }
    #
    # async def _ocr_image(self, image: Image.Image) -> str:
    #     """EasyOCR로 이미지에서 텍스트 추출 (self.reader 사용, numpy 배열 필요)"""
    #     import numpy as np
    #     if image.mode != "RGB":
    #         image = image.convert("RGB")
    #     img_array = np.array(image)
    #     loop = asyncio.get_event_loop()
    #     result = await loop.run_in_executor(
    #         None,
    #         lambda: self.reader.readtext(img_array, detail=0, paragraph=True),
    #     )
    #     return "\n".join(result) if isinstance(result, list) else str(result)
