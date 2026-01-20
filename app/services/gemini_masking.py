"""
Gemini 3 Flash Preview를 사용한 얼굴 PII 마스킹 서비스

Gemini 3 Flash Preview의 bounding box detection을 사용하여 얼굴을 감지하고 원형으로 마스킹합니다.
실패 시 OpenCV Haar Cascade를 fallback으로 사용합니다.
"""

import io
import os
import tempfile
import base64
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import httpx
from PIL import Image, ImageDraw
import pdf2image
import logging
import google.genai as genai
from google.genai import types

logger = logging.getLogger(__name__)

# Presidio imports
try:
    from presidio_image_redactor import ImageAnalyzerEngine, ImageRedactorEngine
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available. Text PII detection will be disabled.")


class GeminiPIIMaskingService:
    """Gemini 3 Flash Preview를 사용한 얼굴 PII 마스킹 서비스 (OpenCV fallback 포함)"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Gemini API 키 (환경변수 GOOGLE_API_KEY 또는 GEMINI_API_KEY 사용 가능)
        """
        # Gemini 클라이언트 초기화
        if api_key:
            genai.configure(api_key=api_key)

        # Presidio 초기화 (텍스트 PII 감지용)
        if PRESIDIO_AVAILABLE:
            try:
                self.presidio_analyzer = AnalyzerEngine()
                self.presidio_image_analyzer = ImageAnalyzerEngine(self.presidio_analyzer)
                self.presidio_image_redactor = ImageRedactorEngine(self.presidio_image_analyzer)
                logger.info("Presidio initialized for text PII detection")
            except Exception as e:
                logger.error(f"Failed to initialize Presidio: {e}")
                self.presidio_analyzer = None
                self.presidio_image_analyzer = None
                self.presidio_image_redactor = None
        else:
            self.presidio_analyzer = None
            self.presidio_image_analyzer = None
            self.presidio_image_redactor = None

    async def download_file(self, file_url: str) -> bytes:
        """
        URL에서 파일 다운로드 (data: URL도 지원)

        Args:
            file_url: 파일 URL (http(s):// 또는 data:)

        Returns:
            파일 바이트 데이터
        """
        # data: URL 처리
        if file_url.startswith('data:'):
            # data:image/png;base64,... 형식
            try:
                header, encoded = file_url.split(',', 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"Failed to decode data URL: {e}")
                raise ValueError("Invalid data URL format")

        # HTTP(S) URL 처리
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=30.0)
            response.raise_for_status()
            return response.content

    def pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
        """
        PDF를 이미지 리스트로 변환

        Args:
            pdf_bytes: PDF 파일 바이트 데이터
            dpi: 이미지 해상도 (기본값: 200)

        Returns:
            이미지 리스트
        """
        # PIL의 decompression bomb 제한 해제 (큰 이미지 처리 허용)
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

    def detect_faces_with_opencv(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        OpenCV Haar Cascade를 사용한 얼굴 감지 (Gemini 실패 시 fallback)

        Args:
            image: PIL Image 객체

        Returns:
            감지된 얼굴 정보 리스트
        """
        try:
            import cv2
            import numpy as np

            logger.info("Using OpenCV Haar Cascade for face detection (fallback)")

            # PIL 이미지를 OpenCV 형식으로 변환
            img_array = np.array(image)

            # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array

            # 그레이스케일 변환
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Haar Cascade 분류기 로드
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            if face_cascade.empty():
                logger.error("Failed to load Haar Cascade classifier")
                return []

            # 얼굴 감지
            # scaleFactor: 이미지 크기를 축소하는 비율 (작을수록 정확하지만 느림)
            # minNeighbors: 얼굴로 판단하기 위한 최소 이웃 개수 (높을수록 정확하지만 놓칠 수 있음)
            # minSize: 감지할 최소 얼굴 크기
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            logger.info(f"OpenCV detected {len(faces)} face(s)")

            detections = []
            for idx, (x, y, w, h) in enumerate(faces):
                logger.info(f"OpenCV Face {idx+1}: x={x}, y={y}, w={w}, h={h}")
                detections.append({
                    "type": "face",
                    "coordinates": [int(x), int(y), int(x + w), int(y + h)],
                    "confidence": 0.85,  # OpenCV는 confidence를 제공하지 않으므로 고정값 사용
                    "text": ""
                })

            logger.info(f"✅ OpenCV fallback detected {len(detections)} face(s)")
            return detections

        except ImportError as ie:
            logger.error(f"OpenCV not installed: {ie}")
            logger.error("Install with: pip install opencv-python>=4.8.0")
            return []
        except Exception as e:
            logger.error(f"Error in OpenCV face detection: {e}", exc_info=True)
            return []

    async def detect_faces_with_gemini(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Gemini Vision API의 bounding_box_detection을 사용하여 얼굴 감지

        Args:
            image: PIL Image 객체

        Returns:
            감지된 얼굴 정보 리스트
        """
        try:
            import google.genai
            from google.genai import types

            logger.info("Using Gemini bounding_box_detection for face detection")

            # 이미지를 바이트로 변환
            buffered = io.BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()

            # Gemini 클라이언트 초기화 (GEMINI_API_KEY 또는 GOOGLE_API_KEY)
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
                logger.info("Falling back to OpenCV due to missing API key")
                return await asyncio.to_thread(self.detect_faces_with_opencv, image)

            client = google.genai.Client(api_key=api_key)

            # Gemini에 얼굴 감지 요청 (간단한 프롬프트)
            prompt = types.Part.from_text(
                text="""Detect all human faces (profile photos, ID photos) in this image.
For each face found, return the bounding box coordinates.

Return ONLY a JSON object in this exact format:
{
  "faces": [
    {"x": <left>, "y": <top>, "width": <width>, "height": <height>}
  ]
}

If no faces: {"faces": []}"""
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt
                    ]
                )
            ]

            # JSON 응답 요청
            generation_config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )

            response = await asyncio.to_thread(
                client.models.generate_content,
                model='gemini-3-flash-preview',
                contents=contents,
                config=generation_config
            )

            # 응답 파싱
            import json
            import re

            text = response.text.strip()
            logger.info(f"Gemini bounding_box response: {text[:500]}")

            # JSON 추출
            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                logger.warning("No JSON found in Gemini response, falling back to OpenCV")
                return await asyncio.to_thread(self.detect_faces_with_opencv, image)

            result = json.loads(json_match.group())
            faces = result.get('faces', [])

            detections = []
            for idx, face in enumerate(faces, 1):
                box = face.get('box', {})
                x = box.get('x', 0)
                y = box.get('y', 0)
                w = box.get('width', 0)
                h = box.get('height', 0)
                confidence = face.get('confidence', 0.9)

                if w > 0 and h > 0:
                    logger.info(f"Gemini Face {idx}: x={x}, y={y}, w={w}, h={h}, conf={confidence}")
                    detections.append({
                        "type": "face",
                        "coordinates": [int(x), int(y), int(x + w), int(y + h)],
                        "confidence": confidence,
                        "text": ""
                    })

            if len(detections) == 0:
                logger.warning("Gemini detected no faces, falling back to OpenCV")
                return await asyncio.to_thread(self.detect_faces_with_opencv, image)

            logger.info(f"✅ Gemini detected {len(detections)} face(s)")
            return detections

        except Exception as e:
            logger.error(f"Error in Gemini face detection: {e}", exc_info=True)
            logger.info("Falling back to OpenCV due to error")
            return await asyncio.to_thread(self.detect_faces_with_opencv, image)

    def extract_pii_from_pdf_page(self, pdf_bytes: bytes, page_num: int = 0) -> List[Dict[str, Any]]:
        """
        OCR + Regex로 PDF 이미지에서 텍스트 PII 감지
        (전화번호, 이메일, URL, 한글 이름, 주소, 대학교명)

        Args:
            pdf_bytes: PDF 바이트 데이터
            page_num: 페이지 번호 (0-indexed)

        Returns:
            감지된 PII 정보 리스트
        """
        try:
            import pytesseract
            import re

            detections = []

            # PDF를 이미지로 변환
            images = self.pdf_to_images(pdf_bytes)
            if page_num >= len(images):
                logger.warning(f"Page {page_num} out of range (total pages: {len(images)})")
                return []

            image = images[page_num]

            # PII 패턴 정의
            phone_pattern = re.compile(r'\b0\d{1,2}-?\d{3,4}-?\d{4}\b')  # 한국 전화번호
            email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')  # 이메일
            url_pattern = re.compile(r'https?://[^\s]+')  # URL
            korean_name_pattern = re.compile(r'^[가-힣]{2,4}$')  # 2-4글자 한글 이름
            address_pattern = re.compile(r'[가-힣]+시\s*[가-힣]+구')  # 서울특별시 강남구 형식
            university_pattern = re.compile(r'[가-힣]+대학교')  # 대학교명

            # 학과명 패턴 추가
            major_pattern = re.compile(r'[가-힣]+학과')  # XX학과

            # OCR로 텍스트 및 좌표 추출 (한글 + 영어)
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='kor+eng')

            logger.info(f"OCR extracted {len(ocr_data['text'])} text boxes from PDF page {page_num}")

            # 첫 번째 한글 이름 찾기
            found_name = False

            # 각 OCR 박스에서 PII 패턴 검색
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i]
                if not text.strip():
                    continue

                conf = int(ocr_data['conf'][i])
                if conf < 30:  # 신뢰도 낮은 텍스트 제외
                    continue

                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]

                pii_type = None

                # 전화번호 체크
                if phone_pattern.search(text):
                    pii_type = "phone_number"
                # 이메일 체크
                elif email_pattern.search(text):
                    pii_type = "email_address"
                # URL 체크
                elif url_pattern.search(text):
                    pii_type = "url"
                # 대학교명 체크
                elif university_pattern.search(text):
                    pii_type = "university"
                # 학과명 체크
                elif major_pattern.search(text):
                    pii_type = "major"
                # 주소 체크
                elif address_pattern.search(text):
                    pii_type = "address"
                # 한글 이름 체크 (상단 영역, 첫 번째 한글만)
                elif not found_name and korean_name_pattern.match(text) and y < 400:
                    pii_type = "name"
                    found_name = True

                if pii_type:
                    detection = {
                        "type": pii_type,
                        "coordinates": [x, y, x + w, y + h],
                        "confidence": conf / 100.0,
                        "text": text
                    }
                    detections.append(detection)
                    logger.info(f"Found {pii_type.upper()}: '{text}' at [{x}, {y}, {x+w}, {y+h}] (conf: {conf})")

            logger.info(f"OCR + Regex found {len(detections)} PII items")
            return detections

        except Exception as e:
            logger.error(f"Error extracting PII from PDF with OCR: {e}", exc_info=True)
            return []

    async def detect_text_pii_with_pdfplumber(self, pdf_bytes: bytes, page_num: int = 0) -> List[Dict[str, Any]]:
        """
        비동기 래퍼: pdfplumber + Presidio로 텍스트 PII 감지

        Args:
            pdf_bytes: PDF 바이트 데이터
            page_num: 페이지 번호

        Returns:
            감지된 PII 정보 리스트
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_pii_from_pdf_page, pdf_bytes, page_num)

    async def detect_text_pii_with_presidio(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Presidio를 사용하여 이미지에서 텍스트 PII 감지

        Args:
            image: PIL Image 객체

        Returns:
            감지된 PII 정보 리스트
        """
        if not self.presidio_image_redactor:
            logger.warning("Presidio not available, skipping text PII detection")
            return []

        try:
            import asyncio
            loop = asyncio.get_event_loop()

            def _detect():
                # Presidio로 PII 감지
                # ImageRedactorEngine은 bounding box를 제공하지 않으므로
                # 대신 AnalyzerEngine으로 OCR + 텍스트 분석
                import pytesseract

                # OCR로 텍스트 추출 (한글 + 영어)
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='kor+eng')

                detections = []
                n_boxes = len(ocr_data['text'])

                # 전체 텍스트 추출 (문맥 분석용)
                full_text = " ".join([ocr_data['text'][i] for i in range(n_boxes) if ocr_data['text'][i].strip()])

                # 전체 텍스트에서 PII 분석 (문맥 기반)
                all_results = self.presidio_analyzer.analyze(
                    text=full_text,
                    language='en',
                    entities=[
                        "PHONE_NUMBER",
                        "EMAIL_ADDRESS",
                        "PERSON",
                        "LOCATION",
                        "US_SSN",  # 주민등록번호 패턴과 유사
                        "CREDIT_CARD",
                        "US_DRIVER_LICENSE",
                        "US_PASSPORT",
                        "NRP",  # Named Entity Recognition Patterns
                    ]
                )

                # 감지된 PII 텍스트 집합
                pii_texts = set()
                for result in all_results:
                    pii_text = full_text[result.start:result.end]
                    pii_texts.add(pii_text.lower().strip())
                    logger.info(f"Presidio found in context: {result.entity_type} = '{pii_text}'")

                # OCR 박스와 매칭
                for i in range(n_boxes):
                    text = ocr_data['text'][i]
                    if not text.strip():
                        continue

                    # 개별 단어도 체크
                    word_results = self.presidio_analyzer.analyze(
                        text=text,
                        language='en',
                        entities=[
                            "PHONE_NUMBER",
                            "EMAIL_ADDRESS",
                            "PERSON",
                            "LOCATION",
                            "US_SSN",
                            "CREDIT_CARD",
                        ]
                    )

                    # 문맥에서 찾은 PII와 매칭되는지 확인
                    is_pii = False
                    entity_type = "unknown"
                    confidence = 0.0

                    if word_results:
                        is_pii = True
                        entity_type = word_results[0].entity_type
                        confidence = word_results[0].score
                    else:
                        # 문맥에서 감지된 PII에 포함되는지 확인
                        text_lower = text.lower().strip()
                        for pii_text in pii_texts:
                            if text_lower in pii_text or pii_text in text_lower:
                                is_pii = True
                                entity_type = "PERSON"  # 기본값
                                confidence = 0.7
                                break

                    if is_pii:
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]

                        detection = {
                            "type": entity_type.lower(),
                            "coordinates": [x, y, x + w, y + h],
                            "confidence": confidence,
                            "text": text
                        }
                        detections.append(detection)
                        logger.info(f"Presidio detected {entity_type}: {text} at [{x}, {y}, {x+w}, {y+h}]")

                return detections

            result = await loop.run_in_executor(None, _detect)
            logger.info(f"Presidio detected {len(result)} text PII items")
            return result

        except Exception as e:
            logger.error(f"Error in Presidio PII detection: {e}", exc_info=True)
            return []

    def mask_image_with_detections(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        감지된 PII 영역을 마스킹 처리

        Args:
            image: PIL Image 객체
            detections: 감지된 PII 정보 리스트

        Returns:
            마스킹된 이미지
        """
        logger.info(f"=== MASKING START ===")
        logger.info(f"Image size: {image.width}x{image.height}")
        logger.info(f"Detections to mask: {len(detections)}")

        masked = image.copy()
        draw = ImageDraw.Draw(masked)

        if len(detections) == 0:
            logger.warning("⚠ No faces to mask - returning original image")
            return masked

        for idx, detection in enumerate(detections):
            coords = detection['coordinates']
            pii_type = detection.get('type', 'unknown')

            # 좌표 추출
            x1, y1, x2, y2 = coords

            logger.info(f"{pii_type.upper()} {idx+1}/{len(detections)}:")
            logger.info(f"  Original coords: [{x1}, {y1}, {x2}, {y2}]")
            logger.info(f"  Size: {x2-x1}x{y2-y1}")

            # PII 타입별 마스킹 처리
            if pii_type == 'face':
                # 얼굴은 원형으로 마스킹
                # 중심점 계산
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 반지름 계산 (박스의 긴 쪽을 기준으로)
                radius = max(x2 - x1, y2 - y1) // 2
                # 약간 크게 마스킹 (패딩 효과)
                radius = int(radius * 1.15)

                # 원형 마스킹
                draw.ellipse(
                    [center_x - radius, center_y - radius,
                     center_x + radius, center_y + radius],
                    fill='black',
                    outline='black'
                )
                logger.info(f"  Circle: center=({center_x}, {center_y}), radius={radius}")
            else:
                # 텍스트 PII는 사각형으로 마스킹
                padding = 10
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(image.width, x2 + padding)
                y2_pad = min(image.height, y2 + padding)

                draw.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], fill='black', outline='black')
                logger.info(f"  Rectangle with padding ({padding}px)")

            logger.info(f"  ✓ Masked successfully")

        logger.info(f"=== MASKING COMPLETE: {len(detections)} face(s) masked ===")
        return masked

    def images_to_pdf(self, images: List[Image.Image]) -> bytes:
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
            if img.mode != 'RGB':
                img = img.convert('RGB')
            rgb_images.append(img)

        # PDF로 저장
        rgb_images[0].save(
            output,
            format='PDF',
            save_all=True,
            append_images=rgb_images[1:] if len(rgb_images) > 1 else []
        )

        return output.getvalue()

    async def mask_image_file(
        self,
        file_url: str
    ) -> Tuple[bytes, bytes, List[Dict[str, Any]]]:
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

        # 2. Gemini로 얼굴 감지
        logger.info("Detecting faces with Gemini...")
        detections = await self.detect_faces_with_gemini(image)

        # 3. 마스킹 처리
        logger.info(f"Masking {len(detections)} PII items...")
        masked_image = self.mask_image_with_detections(image, detections)

        # 4. 바이트로 변환
        output = io.BytesIO()
        masked_image.save(output, format='PNG')
        masked_bytes = output.getvalue()

        # 5. 썸네일 생성
        thumbnail = masked_image.copy()
        thumbnail.thumbnail((300, 400))
        thumb_io = io.BytesIO()
        thumbnail.save(thumb_io, format='PNG')
        thumbnail_bytes = thumb_io.getvalue()

        # 페이지 정보 추가
        for det in detections:
            det['page'] = 1

        logger.info(f"Masking complete: {len(detections)} PII items detected")

        return masked_bytes, thumbnail_bytes, detections

    async def mask_pdf(
        self,
        file_url: str
    ) -> Tuple[bytes, bytes, List[Dict[str, Any]]]:
        """
        PDF 파일 마스킹 처리 (얼굴 감지 전용)

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
                # 첫 페이지만 얼굴 감지 및 마스킹
                logger.info(f"Masking page 1/{len(images)} - Gemini face detection only")

                # Gemini로 얼굴 감지
                face_detections = await self.detect_faces_with_gemini(image)
                logger.info(f"Gemini detected {len(face_detections)} face(s)")

                # 얼굴만 마스킹
                detections = face_detections

                # 마스킹 처리
                if len(detections) > 0:
                    masked_img = self.mask_image_with_detections(image, detections)
                    masked_images.append(masked_img)
                else:
                    logger.info("No faces detected, using original image")
                    masked_images.append(image)

                # 페이지 정보 추가
                for det in detections:
                    det['page'] = page_num
                all_detections.extend(detections)

                logger.info(f"First page: detected {len(detections)} face(s)")
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
        thumbnail.save(thumb_io, format='PNG')
        thumbnail_bytes = thumb_io.getvalue()

        logger.info(f"Masking complete: {len(all_detections)} PII items detected")

        return masked_pdf_bytes, thumbnail_bytes, all_detections


# 전역 서비스 인스턴스
_gemini_service: Optional[GeminiPIIMaskingService] = None


def get_gemini_masking_service(api_key: Optional[str] = None) -> GeminiPIIMaskingService:
    """Gemini PII 마스킹 서비스 싱글톤 인스턴스 반환"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiPIIMaskingService(api_key=api_key)
    return _gemini_service
