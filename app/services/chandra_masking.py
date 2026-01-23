"""
Chandra 모델을 사용한 OCR + PII 마스킹 서비스

datalab-to/chandra 모델을 사용하여 이미지에서 텍스트를 OCR하고 개인정보를 감지하여 마스킹합니다.
"""

import asyncio
import base64
import io
import logging
import os
import re
import tempfile
from typing import Any

import httpx
import pdf2image
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Chandra 모델 관련 import
try:
    from transformers import AutoModel, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Chandra model will be disabled.")

try:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown

    CHANDRA_AVAILABLE = True
except ImportError:
    CHANDRA_AVAILABLE = False
    logger.warning("Chandra package not available. Please install from datalab-to/chandra")


class ChandraPIIMaskingService:
    """Chandra 모델을 사용한 PII 마스킹 서비스"""

    def __init__(self, model_name: str = "datalab-to/chandra", device: str = "cuda"):
        """
        Args:
            model_name: Chandra 모델 이름
            device: 디바이스 (cuda/cpu)
        """
        if not TRANSFORMERS_AVAILABLE or not CHANDRA_AVAILABLE:
            raise ImportError(
                "Chandra dependencies not available. "
                "Please install: pip install transformers chandra"
            )

        # Chandra 모델 초기화
        logger.info(f"Loading Chandra model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)

        # 디바이스 설정
        if device == "cuda" and not self._is_cuda_available():
            logger.warning("CUDA not available, using CPU instead")
            device = "cpu"

        if device == "cuda":
            self.model = self.model.cuda()

        self.model.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        logger.info(f"Chandra model loaded on {device}")

    def _is_cuda_available(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def download_file(self, file_url: str) -> bytes:
        """
        URL에서 파일 다운로드 (data: URL도 지원)

        Args:
            file_url: 파일 URL (http(s):// 또는 data:)

        Returns:
            파일 바이트 데이터
        """
        # data: URL 처리
        if file_url.startswith("data:"):
            try:
                header, encoded = file_url.split(",", 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"Failed to decode data URL: {e}")
                raise ValueError("Invalid data URL format")

        # HTTP(S) URL 처리
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=30.0)
            response.raise_for_status()
            return response.content

    def pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
        """
        PDF를 이미지 리스트로 변환

        Args:
            pdf_bytes: PDF 파일 바이트 데이터
            dpi: 이미지 해상도 (기본값: 200)

        Returns:
            이미지 리스트
        """
        from PIL import Image as PILImage

        PILImage.MAX_IMAGE_PIXELS = None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            images = pdf2image.convert_from_path(tmp_path, dpi=dpi)
            logger.info(f"Converted PDF to {len(images)} images")
            return images
        finally:
            os.unlink(tmp_path)

    async def detect_pii_with_chandra(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Chandra 모델을 사용하여 이미지에서 OCR + PII 감지

        Args:
            image: PIL Image 객체

        Returns:
            감지된 PII 정보 리스트
        """
        try:
            logger.info("Processing image with Chandra model (OCR + Layout analysis)")

            # BatchInputItem 생성
            batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]

            # 동기 함수를 비동기로 실행
            def _run_chandra():
                result = generate_hf(batch, self.model)[0]
                markdown = parse_markdown(result.raw)
                return result, markdown

            result, markdown = await asyncio.to_thread(_run_chandra)

            logger.info(f"Chandra OCR result:\n{markdown}")

            # Markdown에서 텍스트 추출 및 PII 감지
            detections = self._extract_pii_from_markdown(markdown, image)

            logger.info(f"Chandra detected {len(detections)} PII items")
            return detections

        except Exception as e:
            logger.error(f"Error in Chandra PII detection: {e}", exc_info=True)
            return []

    def _extract_pii_from_markdown(self, markdown: str, image: Image.Image) -> list[dict[str, Any]]:
        """
        Chandra의 Markdown 결과에서 텍스트 PII 감지 (얼굴 제외)

        Args:
            markdown: Chandra의 OCR 결과 (Markdown)
            image: 원본 이미지

        Returns:
            감지된 PII 정보 리스트
        """
        detections = []

        # 텍스트 PII 패턴 정의 (한국어 중심)
        patterns = {
            "phone_number": re.compile(r"\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b"),  # 한국 전화번호
            "email_address": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),  # 이메일
            "url": re.compile(r"https?://[^\s]+"),  # URL
            "name": re.compile(r"\b[가-힣]{2,4}\b"),  # 2-4글자 한글 이름
            "address": re.compile(r"[가-힣]+시\s*[가-힣]+구"),  # 주소 (시/구)
            "university": re.compile(r"[가-힣]+대학교"),  # 대학교명
            "major": re.compile(r"[가-힣]+학과"),  # 학과명
        }

        # Markdown에서 텍스트 추출
        lines = markdown.split("\n")
        found_name = False
        w, h = image.size

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):  # 빈 줄이나 헤더 제외
                continue

            # 각 패턴 체크
            for pii_type, pattern in patterns.items():
                # 이름은 첫 번째만 감지 (상단 영역)
                if pii_type == "name" and found_name:
                    continue

                matches = pattern.finditer(line)
                for match in matches:
                    text = match.group()

                    # 라인 번호를 기반으로 대략적인 y 좌표 계산
                    y_start = int((line_idx / max(len(lines), 1)) * h)
                    y_end = min(y_start + (h // 20), h)  # 각 라인 높이를 전체의 5% 정도로 가정

                    # x 좌표는 텍스트 위치에 비례하여 계산
                    text_pos_ratio = match.start() / max(len(line), 1)
                    x_start = int(text_pos_ratio * w)
                    x_end = min(x_start + int(len(text) * (w // 50)), w)  # 문자당 너비 추정

                    detection = {
                        "type": pii_type,
                        "coordinates": [x_start, y_start, x_end, y_end],
                        "confidence": 0.85,
                        "text": text,
                    }
                    detections.append(detection)
                    logger.info(f"Found {pii_type.upper()}: '{text}' at line {line_idx}")

                    if pii_type == "name" and line_idx < len(lines) // 3:  # 상단 1/3 영역에서만
                        found_name = True

        return detections

    def mask_image_with_detections(
        self, image: Image.Image, detections: list[dict[str, Any]]
    ) -> Image.Image:
        """
        감지된 PII 영역을 마스킹 처리

        Args:
            image: PIL Image 객체
            detections: 감지된 PII 정보 리스트

        Returns:
            마스킹된 이미지
        """
        logger.info("=== MASKING START ===")
        logger.info(f"Image size: {image.width}x{image.height}")
        logger.info(f"Detections to mask: {len(detections)}")

        masked = image.copy()
        draw = ImageDraw.Draw(masked)

        if len(detections) == 0:
            logger.warning("⚠ No PII to mask - returning original image")
            return masked

        for idx, detection in enumerate(detections):
            coords = detection["coordinates"]
            pii_type = detection.get("type", "unknown")

            x1, y1, x2, y2 = coords

            logger.info(f"{pii_type.upper()} {idx+1}/{len(detections)}:")
            logger.info(f"  Original coords: [{x1}, {y1}, {x2}, {y2}]")

            # 패딩 추가
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)

            logger.info(f"  With padding: [{x1}, {y1}, {x2}, {y2}]")

            # 검은 사각형으로 마스킹
            draw.rectangle([x1, y1, x2, y2], fill="black", outline="black")
            logger.info("  ✓ Masked successfully")

        logger.info(f"=== MASKING COMPLETE: {len(detections)} PII masked ===")
        return masked

    def images_to_pdf(self, images: list[Image.Image]) -> bytes:
        """
        이미지 리스트를 PDF로 변환

        Args:
            images: PIL Image 리스트

        Returns:
            PDF 바이트 데이터
        """
        if not images:
            raise ValueError("No images to convert to PDF")

        output = io.BytesIO()

        # RGB 모드로 변환
        rgb_images = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            rgb_images.append(img)

        # PDF로 저장
        rgb_images[0].save(
            output,
            format="PDF",
            save_all=True,
            append_images=rgb_images[1:] if len(rgb_images) > 1 else [],
        )

        return output.getvalue()

    async def mask_image_file(self, file_url: str) -> tuple[bytes, bytes, list[dict[str, Any]]]:
        """
        이미지 파일 마스킹 처리

        Args:
            file_url: 이미지 파일 URL

        Returns:
            (마스킹된 이미지 바이트, 썸네일 바이트, PII 감지 정보)
        """
        # 1. 이미지 다운로드
        logger.info(f"Downloading image from {file_url}")
        image_bytes = await self.download_file(file_url)
        image = Image.open(io.BytesIO(image_bytes))

        # 2. Chandra로 OCR + PII 감지
        logger.info("Detecting PII with Chandra model...")
        detections = await self.detect_pii_with_chandra(image)

        # 3. 마스킹 처리
        logger.info(f"Masking {len(detections)} PII items...")
        masked_image = self.mask_image_with_detections(image, detections)

        # 4. 바이트로 변환
        output = io.BytesIO()
        masked_image.save(output, format="PNG")
        masked_bytes = output.getvalue()

        # 5. 썸네일 생성
        thumbnail = masked_image.copy()
        thumbnail.thumbnail((300, 400))
        thumb_io = io.BytesIO()
        thumbnail.save(thumb_io, format="PNG")
        thumbnail_bytes = thumb_io.getvalue()

        # 페이지 정보 추가
        for det in detections:
            det["page"] = 1

        logger.info(f"Masking complete: {len(detections)} PII items detected")

        return masked_bytes, thumbnail_bytes, detections

    async def mask_pdf(self, file_url: str) -> tuple[bytes, bytes, list[dict[str, Any]]]:
        """
        PDF 파일 마스킹 처리

        Args:
            file_url: PDF 파일 URL

        Returns:
            (마스킹된 PDF 바이트, 썸네일 바이트, 전체 페이지의 PII 감지 정보)
        """
        # 1. PDF 다운로드
        logger.info(f"Downloading PDF from {file_url}")
        pdf_bytes = await self.download_file(file_url)

        # 2. PDF를 이미지로 변환
        logger.info("Converting PDF to images")
        images = self.pdf_to_images(pdf_bytes)

        # 3. 첫 페이지만 마스킹 (API 비용 절감)
        masked_images = []
        all_detections = []

        for page_num, image in enumerate(images, start=1):
            if page_num == 1:
                # 첫 페이지만 Chandra로 PII 감지 및 마스킹
                logger.info(f"Masking page 1/{len(images)} - Chandra OCR + PII detection")

                detections = await self.detect_pii_with_chandra(image)
                logger.info(f"Chandra detected {len(detections)} PII items")

                # 마스킹 처리
                if len(detections) > 0:
                    masked_img = self.mask_image_with_detections(image, detections)
                    masked_images.append(masked_img)
                else:
                    logger.info("No PII detected, using original image")
                    masked_images.append(image)

                # 페이지 정보 추가
                for det in detections:
                    det["page"] = page_num
                all_detections.extend(detections)

                logger.info(f"First page: detected {len(detections)} PII items")
            else:
                # 나머지 페이지는 마스킹 없이 원본 유지
                logger.info(f"Page {page_num}/{len(images)} - keeping original (no masking)")
                masked_images.append(image)

        # 4. 마스킹된 이미지를 PDF로 변환
        logger.info("Converting masked images back to PDF")
        masked_pdf_bytes = self.images_to_pdf(masked_images)

        # 5. 썸네일 생성 (첫 페이지)
        logger.info("Creating thumbnail")
        thumbnail = masked_images[0].copy()
        thumbnail.thumbnail((300, 400))
        thumb_io = io.BytesIO()
        thumbnail.save(thumb_io, format="PNG")
        thumbnail_bytes = thumb_io.getvalue()

        logger.info(f"Masking complete: {len(all_detections)} PII items detected")

        return masked_pdf_bytes, thumbnail_bytes, all_detections


# 전역 서비스 인스턴스
_chandra_service: ChandraPIIMaskingService | None = None


def get_chandra_masking_service(
    model_name: str = "datalab-to/chandra", device: str = "cuda"
) -> ChandraPIIMaskingService:
    """Chandra PII 마스킹 서비스 싱글톤 인스턴스 반환"""
    global _chandra_service
    if _chandra_service is None:
        _chandra_service = ChandraPIIMaskingService(model_name=model_name, device=device)
    return _chandra_service
