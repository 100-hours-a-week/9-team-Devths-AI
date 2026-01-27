"""
PII Masking Service using HuggingFace Space

이 서비스는 PDF 파일에서 개인정보를 감지하고 마스킹 처리합니다.
"""

import io
import logging
import os
import tempfile
from typing import Any

import httpx
import numpy as np
import pdf2image
from gradio_client import Client, handle_file
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# 테스트용 함수들 (현재 사용되지 않음)
# def plt_imshow(title='image', img=None, figsize=(8 ,5)):
#     plt.figure(figsize=figsize)
#     if isinstance(img, list):
#         if isinstance(title, list):
#             titles = title
#         else:
#             titles = [title] * len(img)
#         for idx, img_item in enumerate(img):
#             if len(img_item.shape) <= 2:
#                 rgbImg = cv2.cvtColor(img_item, cv2.COLOR_GRAY2RGB)
#             else:
#                 rgbImg = cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB)
#             plt.subplot(1, len(img), idx + 1), plt.imshow(rgbImg)
#             plt.title(titles[idx])
#             plt.xticks([]), plt.yticks([])
#         plt.show()
#     else:
#         if len(img.shape) < 3:
#             rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         else:
#             rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.imshow(rgbImg)
#         plt.title(title)
#         plt.xticks([]), plt.yticks([])
#         plt.show()
#
# def make_scan_image(org_image, image, width, ksize=(5,5), min_threshold=75, max_threshold=200):
#   image_list_title = []
#   image_list = []
#   image = imutils.resize(image, width=width)
#   ratio = org_image.shape[1] / float(image.shape[1])
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   blurred = cv2.GaussianBlur(gray, ksize, 0)
#   edged = cv2.Canny(blurred, min_threshold, max_threshold)
#   image_list_title = ['gray', 'blurred', 'edged']
#   image_list = [gray, blurred, edged]
#   cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   cnts = imutils.grab_contours(cnts)
#   cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#   findCnt = None
#   for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#     if len(approx) == 4:
#       findCnt = approx
#       break
#   if findCnt is None:
#     raise Exception("Could not find outline.")
#   output = image.copy()
#   cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
#   image_list_title.append("Outline")
#   image_list.append(output)
#   transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)
#   plt_imshow(image_list_title, image_list)
#   plt_imshow("Transform", transform_image)
#   return transform_image

def putText(cv_img, text, x, y, color=(0, 0, 0), font_size=22):
  # Colab이 아닌 Local에서 수행 시에는 gulim.ttc 를 사용하면 됩니다.
  # font = ImageFont.truetype("fonts/gulim.ttc", font_size)
  font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf', font_size)
  img = Image.fromarray(cv_img)

  draw = ImageDraw.Draw(img)
  draw.text((x, y), text, font=font, fill=color)

  cv_img = np.array(img)

  return cv_img


# 테스트 코드 (주석 처리)
# image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
# org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
# plt_imshow("orignal image", org_image)
# business_card_image = make_scan_image(org_image, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)
# langs = ['ko', 'en']
# print("[INFO] OCR'ing input image...")
# reader = Reader(lang_list=langs, gpu=True)
# results = reader.readtext(business_card_image)
# for (bbox, text, prob) in results:
#   print(f"[INFO] {prob:.4f}: {text}")
#   (tl, tr, br, bl) = bbox
#   tl = (int(tl[0]), int(tl[1]))
#   tr = (int(tr[0]), int(tr[1]))
#   br = (int(br[0]), int(br[1]))
#   bl = (int(bl[0]), int(bl[1]))
#   # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
#   cv2.rectangle(business_card_image, tl, br, (0, 255, 0), 2)
#   business_card_image = putText(business_card_image, text, tl[0], tl[1] - 60, (0, 255, 0), 50)
#   plt_imshow("Image", business_card_image, figsize=(16,10))
# simple_results = reader.readtext(business_card_image, detail = 0)


class PIIMaskingService:
    """PII 마스킹 서비스"""

    def __init__(self, space_name: str = "LibrAI/PII-Redactor"):
        """
        Args:
            space_name: HuggingFace Space 이름 (예: "LibrAI/PII-Redactor")
        """
        self.space_name = space_name
        self.client = None

    def _initialize_client(self):
        """Gradio Client 초기화 (필요할 때만)"""
        if self.client is None:
            try:
                self.client = Client(self.space_name)
                logger.info(f"Gradio client initialized for {self.space_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gradio client: {e}")
                raise

    async def download_file(self, file_url: str) -> bytes:
        """
        URL에서 파일 다운로드

        Args:
            file_url: 파일 URL

        Returns:
            파일 바이트 데이터
        """
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
        # PDF 바이트를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            # PDF를 이미지로 변환
            images = pdf2image.convert_from_path(tmp_path, dpi=dpi)
            logger.info(f"Converted PDF to {len(images)} images")
            return images
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_path)

    def mask_image_with_space(self, image: Image.Image) -> tuple[Image.Image, list[dict[str, Any]]]:
        """
        HuggingFace Space를 사용하여 이미지에서 PII 마스킹

        Args:
            image: PIL Image 객체

        Returns:
            (마스킹된 이미지, 감지된 PII 정보 리스트)
        """
        self._initialize_client()

        # 이미지를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file, format="PNG")
            tmp_path = tmp_file.name

        try:
            # Gradio Space 호출
            # 주의: Space의 실제 API 구조에 따라 수정 필요
            result = self.client.predict(
                image=handle_file(tmp_path),
                api_name="/predict"  # Space의 실제 API 엔드포인트에 맞게 수정
            )

            # 결과 파싱 (Space 출력 형식에 따라 다름)
            if isinstance(result, tuple):
                masked_image_path, detections = result
                masked_image = Image.open(masked_image_path)
            else:
                # 결과 형식에 따라 조정
                masked_image = Image.open(result)
                detections = []

            logger.info(f"Masked image with {len(detections)} PII detections")
            return masked_image, detections

        except Exception as e:
            logger.error(f"Error in Space prediction: {e}")
            # Space 사용 실패 시 로컬 마스킹 사용
            return self._fallback_local_masking(image)
        finally:
            os.unlink(tmp_path)

    def _fallback_local_masking(self, image: Image.Image) -> tuple[Image.Image, list[dict[str, Any]]]:
        """
        로컬 PII 감지 및 마스킹 (fallback)

        간단한 패턴 매칭 기반 (실제로는 더 정교한 모델 필요)

        Args:
            image: PIL Image 객체

        Returns:
            (마스킹된 이미지, 감지된 PII 정보 리스트)
        """
        logger.warning("Using fallback local masking (EasyOCR)")

        try:
            from easyocr import Reader
            from PIL import ImageDraw

            # EasyOCR Reader 초기화
            reader = Reader(['ko', 'en'], gpu=False)
            # EasyOCR로 텍스트 및 위치 추출
            img_array = np.array(image)
            results = reader.readtext(img_array)

            # 마스킹할 영역 찾기 (간단한 패턴)
            detections = []
            masked_image = image.copy()
            draw = ImageDraw.Draw(masked_image)

            for (bbox, text, conf) in results:
                if text.strip():
                    # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    (tl, tr, br, bl) = bbox
                    x = int(tl[0])
                    y = int(tl[1])
                    w = int(tr[0] - tl[0])
                    h = int(bl[1] - tl[1])

                    # 전화번호 패턴 감지 (예: 010-1234-5678)
                    if self._is_phone_number(text):
                        # 검은색 박스로 마스킹
                        draw.rectangle([x, y, x+w, y+h], fill='black')
                        detections.append({
                            "type": "phone",
                            "coordinates": [x, y, x+w, y+h],
                            "confidence": conf,
                            "text": text
                        })

                    # 이메일 패턴 감지
                    elif '@' in text and '.' in text:
                        draw.rectangle([x, y, x+w, y+h], fill='black')
                        detections.append({
                            "type": "email",
                            "coordinates": [x, y, x+w, y+h],
                            "confidence": conf,
                            "text": text
                        })

            return masked_image, detections

        except Exception as e:
            logger.error(f"EasyOCR not available for fallback masking: {e}")
            return image, []

    def _is_phone_number(self, text: str) -> bool:
        """전화번호 패턴 감지"""
        import re
        # 한국 전화번호 패턴
        patterns = [
            r'\d{3}-\d{4}-\d{4}',  # 010-1234-5678
            r'\d{3}\d{4}\d{4}',    # 01012345678
            r'\d{2,3}-\d{3,4}-\d{4}',  # 02-1234-5678
        ]
        return any(re.match(pattern, text) for pattern in patterns)

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

        # 첫 번째 이미지를 기준으로 PDF 생성
        output = io.BytesIO()

        # RGB 모드로 변환 (PDF 저장을 위해)
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

    async def mask_pdf(
        self,
        file_url: str,
        use_space: bool = True
    ) -> tuple[bytes, bytes, list[dict[str, Any]]]:
        """
        PDF 파일 마스킹 처리 (전체 플로우)

        Args:
            file_url: PDF 파일 URL
            use_space: HuggingFace Space 사용 여부

        Returns:
            (마스킹된 PDF 바이트, 썸네일 바이트, 전체 페이지의 PII 감지 정보)
        """
        # 1. PDF 다운로드
        logger.info(f"Downloading PDF from {file_url}")
        pdf_bytes = await self.download_file(file_url)

        # 2. PDF를 이미지로 변환
        logger.info("Converting PDF to images")
        images = self.pdf_to_images(pdf_bytes)

        # 3. 각 페이지 마스킹
        masked_images = []
        all_detections = []

        for page_num, image in enumerate(images, start=1):
            logger.info(f"Masking page {page_num}/{len(images)}")

            if use_space:
                try:
                    masked_img, detections = self.mask_image_with_space(image)
                except Exception as e:
                    logger.error(f"Space masking failed for page {page_num}: {e}")
                    masked_img, detections = self._fallback_local_masking(image)
            else:
                masked_img, detections = self._fallback_local_masking(image)

            masked_images.append(masked_img)

            # 페이지 정보 추가
            for det in detections:
                det['page'] = page_num
            all_detections.extend(detections)

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

    async def mask_image_file(
        self,
        file_url: str,
        use_space: bool = True
    ) -> tuple[bytes, bytes, list[dict[str, Any]]]:
        """
        이미지 파일 마스킹 처리

        Args:
            file_url: 이미지 파일 URL
            use_space: HuggingFace Space 사용 여부

        Returns:
            (마스킹된 이미지 바이트, 썸네일 바이트, PII 감지 정보)
        """
        # 1. 이미지 다운로드
        logger.info(f"Downloading image from {file_url}")
        image_bytes = await self.download_file(file_url)
        image = Image.open(io.BytesIO(image_bytes))

        # 2. 마스킹
        if use_space:
            try:
                masked_image, detections = self.mask_image_with_space(image)
            except Exception as e:
                logger.error(f"Space masking failed: {e}")
                masked_image, detections = self._fallback_local_masking(image)
        else:
            masked_image, detections = self._fallback_local_masking(image)

        # 3. 바이트로 변환
        output = io.BytesIO()
        masked_image.save(output, format='PNG')
        masked_bytes = output.getvalue()

        # 4. 썸네일 생성
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


# 전역 서비스 인스턴스
_pii_service: PIIMaskingService | None = None


def get_pii_masking_service(space_name: str = "LibrAI/PII-Redactor") -> PIIMaskingService:
    """PII 마스킹 서비스 싱글톤 인스턴스 반환"""
    global _pii_service
    if _pii_service is None:
        _pii_service = PIIMaskingService(space_name=space_name)
    return _pii_service
