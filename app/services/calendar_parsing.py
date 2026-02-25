"""
Calendar Parsing Service using Gemini Flash API

채용공고 파일/텍스트에서 회사명, 포지션, 일정 정보를 자동으로 추출합니다.
"""

import base64
import io
import logging
import os

import httpx
import pdf2image
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScheduleInfo(BaseModel):
    """일정 정보 (Pydantic 모델)"""

    stage: str = Field(..., description="전형 단계 (예: 서류 마감, 코딩테스트, 1차 면접)")
    date: str = Field(..., description="날짜 (YYYY-MM-DD 형식)")
    time: str | None = Field(None, description="시간 (HH:MM 형식, 정보 없으면 null)")


class JobPostingInfo(BaseModel):
    """채용공고 파싱 결과 (Pydantic 모델)"""

    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션/직무")
    schedules: list[ScheduleInfo] = Field(default_factory=list, description="전형 일정 목록")
    hashtags: list[str] = Field(
        default_factory=list, description="해시태그 (회사명, 직무, 경력 등)"
    )


class CalendarParsingService:
    """Gemini Flash를 사용한 채용공고 파싱 서비스"""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Calendar Parsing Service

        Args:
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        """
        # Configure Gemini API
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"

        logger.info(f"Calendar Parsing Service initialized with model: {self.model_name}")

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
                raise ValueError("Invalid data URL format") from e

        # HTTP(S) URL 처리
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=30.0)
            response.raise_for_status()
            return response.content

    def pdf_to_image(self, pdf_bytes: bytes, dpi: int = 200) -> Image.Image:
        """
        PDF 첫 페이지를 이미지로 변환

        Args:
            pdf_bytes: PDF 파일 바이트 데이터
            dpi: 이미지 해상도 (기본값: 200)

        Returns:
            첫 페이지 이미지
        """
        # 제한 해제
        Image.MAX_IMAGE_PIXELS = None

        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
        if not images:
            raise ValueError("PDF에서 이미지를 추출할 수 없습니다")
        return images[0]

    async def parse_from_image(self, image: Image.Image | bytes) -> JobPostingInfo:
        """
        이미지에서 채용공고 정보 추출

        Args:
            image: PIL Image 객체 또는 이미지 바이트 데이터

        Returns:
            JobPostingInfo: 파싱된 채용공고 정보
        """
        # 이미지가 바이트라면 PIL Image로 변환
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        # 이미지를 Base64로 인코딩
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Gemini API 호출을 위한 프롬프트
        system_prompt = """당신은 채용공고 분석 전문가입니다.
채용공고 이미지에서 다음 정보를 정확하게 추출하세요:

1. **회사명**: 채용 기업명
2. **포지션**: 채용 직무/포지션 (예: 백엔드 개발자, 프론트엔드 개발자, 데이터 분석가 등)
3. **일정 정보**: 서류 마감, 코딩테스트, 면접 등의 전형 일정
   - 날짜는 반드시 YYYY-MM-DD 형식으로 변환
   - 시간은 HH:MM 형식으로 변환 (정보 없으면 null)
4. **해시태그**: 회사명, 직무, 경력(신입/경력), 기술스택 등 관련 키워드를 # 형태로 추출

**중요 규칙:**
- 정보가 명확하지 않으면 추측하지 말고 null이나 빈 리스트를 반환하세요.
- 날짜는 반드시 YYYY-MM-DD 형식으로 변환하세요.
- 해시태그는 # 포함하여 반환하세요.
"""

        user_message = """위 채용공고 이미지를 분석하여 다음 정보를 JSON 형식으로 추출해주세요:
{
  "company": "회사명",
  "position": "포지션",
  "schedules": [
    {
      "stage": "서류 마감",
      "date": "2026-01-15",
      "time": null
    }
  ],
  "hashtags": ["#회사명", "#직무", "#경력"]
}"""

        try:
            # Gemini API 호출
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                            types.Part.from_text(text=user_message),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # 낮은 온도로 일관성 있는 결과
                    response_mime_type="application/json",
                    system_instruction=system_prompt,
                ),
            )

            # 응답 파싱
            result_text = response.text
            logger.info(f"Gemini API response: {result_text}")

            # JSON을 Pydantic 모델로 변환
            import json

            result_dict = json.loads(result_text)
            return JobPostingInfo(**result_dict)

        except Exception as e:
            logger.error(f"Failed to parse image with Gemini API: {e}")
            # 실패 시 기본값 반환
            return JobPostingInfo(
                company="정보 없음",
                position="정보 없음",
                schedules=[],
                hashtags=[],
            )

    async def parse_from_text(self, text: str) -> JobPostingInfo:
        """
        텍스트에서 채용공고 정보 추출

        Args:
            text: 채용공고 텍스트

        Returns:
            JobPostingInfo: 파싱된 채용공고 정보
        """
        system_prompt = """당신은 채용공고 분석 전문가입니다.
채용공고 텍스트에서 다음 정보를 정확하게 추출하세요:

1. **회사명**: 채용 기업명
2. **포지션**: 채용 직무/포지션 (예: 백엔드 개발자, 프론트엔드 개발자, 데이터 분석가 등)
3. **일정 정보**: 서류 마감, 코딩테스트, 면접 등의 전형 일정
   - 날짜는 반드시 YYYY-MM-DD 형식으로 변환
   - 시간은 HH:MM 형식으로 변환 (정보 없으면 null)
4. **해시태그**: 회사명, 직무, 경력(신입/경력), 기술스택 등 관련 키워드를 # 형태로 추출

**중요 규칙:**
- 정보가 명확하지 않으면 추측하지 말고 null이나 빈 리스트를 반환하세요.
- 날짜는 반드시 YYYY-MM-DD 형식으로 변환하세요.
- 해시태그는 # 포함하여 반환하세요.
"""

        user_message = f"""다음 채용공고 텍스트를 분석하여 정보를 JSON 형식으로 추출해주세요:

{text}

JSON 형식:
{{
  "company": "회사명",
  "position": "포지션",
  "schedules": [
    {{
      "stage": "서류 마감",
      "date": "2026-01-15",
      "time": null
    }}
  ],
  "hashtags": ["#회사명", "#직무", "#경력"]
}}"""

        try:
            # Gemini API 호출
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # 낮은 온도로 일관성 있는 결과
                    response_mime_type="application/json",
                    system_instruction=system_prompt,
                ),
            )

            # 응답 파싱
            result_text = response.text
            logger.info(f"Gemini API response: {result_text}")

            # JSON을 Pydantic 모델로 변환
            import json

            result_dict = json.loads(result_text)
            return JobPostingInfo(**result_dict)

        except Exception as e:
            logger.error(f"Failed to parse text with Gemini API: {e}")
            # 실패 시 기본값 반환
            return JobPostingInfo(
                company="정보 없음",
                position="정보 없음",
                schedules=[],
                hashtags=[],
            )

    async def parse_job_posting(
        self, file_url: str | None = None, text: str | None = None
    ) -> JobPostingInfo:
        """
        채용공고 파일 또는 텍스트에서 정보 추출

        Args:
            file_url: 파일 URL (HTTP(S) 또는 data URL)
            text: 채용공고 텍스트

        Returns:
            JobPostingInfo: 파싱된 채용공고 정보
        """
        if file_url:
            # 파일에서 파싱
            logger.info(f"Parsing job posting from file: {file_url}")
            file_bytes = await self.download_file(file_url)

            # PDF 또는 이미지 판단
            if file_url.lower().endswith(".pdf") or file_bytes[:4] == b"%PDF":
                # PDF → 이미지 변환 → 파싱
                logger.info("Converting PDF to image")
                image = self.pdf_to_image(file_bytes)
                return await self.parse_from_image(image)
            else:
                # 이미지 파싱
                logger.info("Parsing image")
                return await self.parse_from_image(file_bytes)

        elif text:
            # 텍스트에서 파싱
            logger.info("Parsing job posting from text")
            return await self.parse_from_text(text)

        else:
            raise ValueError("Either file_url or text must be provided")
