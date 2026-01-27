"""
LLM Service using Gemini Flash API

Provides chat functionality and OCR using Google's Gemini 1.5 Flash model.
"""

import os
import io
import logging
import httpx
from typing import List, Dict, Any, AsyncIterator, Optional
from google import genai
from google.genai import types
from PIL import Image
import pdf2image
import tempfile

from app.utils.langfuse_client import create_generation, trace_llm_call

logger = logging.getLogger(__name__)


class LLMService:
    """LLM Service for chatbot using Gemini Flash"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM Service

        Args:
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        """
        # Configure Gemini API
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"  # Fixed: use gemini-3-flash-preview

        logger.info(f"LLM Service initialized with model: {self.model_name}")

    def _langfuse_trace_and_generation(
        self,
        *,
        trace_name: str,
        generation_name: str,
        input_text: str,
        output_text: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Langfuse trace + generation 기록 헬퍼.

        Langfuse 설정이 없거나 오류가 나더라도 서비스 로직은 중단되지 않도록 no-op로 동작합니다.
        """
        try:
            trace = trace_llm_call(
                name=trace_name,
                user_id=user_id,
                metadata={
                    "model": self.model_name,
                    **(metadata or {}),
                },
            )
            if trace is None:
                return

            create_generation(
                trace=trace,
                name=generation_name,
                model=self.model_name,
                input_text=input_text,
                output_text=output_text,
                metadata=metadata or {},
            )
        except Exception:
            # Langfuse 오류로 본 서비스가 죽지 않게 방어
            return

    async def generate_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from LLM

        Args:
            user_message: User's message
            context: RAG context from VectorDB (optional)
            history: Chat history [{"role": "user", "content": "..."}, ...]
            system_prompt: System instructions (optional)

        Yields:
            Response chunks
        """
        trace = trace_llm_call(
            name="gemini_generate_response",
            user_id=user_id,
            metadata={
                "model": self.model_name,
                "has_context": bool(context),
                "has_history": bool(history),
            },
        )

        full_response = ""
        final_message_for_trace = user_message

        try:
            # Build final user message with context
            final_message = user_message
            if context:
                final_message = f"""관련 정보:
{context}

질문: {user_message}

위 관련 정보를 참고하여 질문에 답변해주세요. 관련 정보가 없으면 일반적인 지식으로 답변해주세요."""
            final_message_for_trace = final_message

            # Create contents using types.Content
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=final_message)]
                )
            ]

            # Create config
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=2048,
                system_instruction=system_prompt if system_prompt else None
            )

            # Generate streaming response
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config
            )

            # Stream chunks
            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # Langfuse generation 기록 (스트리밍 완료 후)
            if trace is not None:
                create_generation(
                    trace=trace,
                    name="gemini_stream",
                    model=self.model_name,
                    input_text=final_message_for_trace,
                    output_text=full_response,
                    metadata={"streaming": True},
                )

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            if trace is not None:
                try:
                    trace["client"].create_event(
                        trace_context={"trace_id": trace["trace_id"]},
                        name="error",
                        level="ERROR",
                        metadata={"error": str(e)},
                    )
                except Exception:
                    pass
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def generate_analysis(
        self,
        resume_text: str,
        posting_text: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate resume and job posting analysis

        Args:
            resume_text: Resume content
            posting_text: Job posting content

        Returns:
            Analysis result as dict
        """
        try:
            prompt = f"""다음 이력서와 채용공고를 분석하여 JSON 형식으로 결과를 제공해주세요.

이력서:
{resume_text}

채용공고:
{posting_text}

다음 JSON 형식으로 분석 결과를 제공해주세요:
{{
  "resume_analysis": {{
    "strengths": ["강점1", "강점2", ...],
    "weaknesses": ["약점1", "약점2", ...],
    "suggestions": ["제안1", "제안2", ...]
  }},
  "posting_analysis": {{
    "company": "회사명",
    "position": "직무",
    "required_skills": ["필수스킬1", "필수스킬2", ...],
    "preferred_skills": ["우대스킬1", "우대스킬2", ...]
  }},
  "matching": {{
    "score": 85,
    "grade": "A",
    "matched_skills": ["매칭된스킬1", "매칭된스킬2", ...],
    "missing_skills": ["부족한스킬1", "부족한스킬2", ...]
  }}
}}"""

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            config = types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            # Extract JSON from response
            import json
            result_text = response.text

            # Try to find JSON in the response
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                self._langfuse_trace_and_generation(
                    trace_name="gemini_generate_analysis",
                    generation_name="gemini_analysis",
                    input_text=prompt,
                    output_text=result_text,
                    user_id=user_id,
                    metadata={"temperature": 0.3, "type": "analysis"},
                )
                return parsed
            else:
                logger.error("No JSON found in analysis response")
                fallback = {
                    "resume_analysis": {
                        "strengths": ["분석 결과를 파싱할 수 없습니다"],
                        "weaknesses": [],
                        "suggestions": []
                    },
                    "posting_analysis": {
                        "company": "알 수 없음",
                        "position": "알 수 없음",
                        "required_skills": [],
                        "preferred_skills": []
                    },
                    "matching": {
                        "score": 0,
                        "grade": "F",
                        "matched_skills": [],
                        "missing_skills": []
                    }
                }
                self._langfuse_trace_and_generation(
                    trace_name="gemini_generate_analysis",
                    generation_name="gemini_analysis",
                    input_text=prompt,
                    output_text=result_text,
                    user_id=user_id,
                    metadata={"temperature": 0.3, "type": "analysis", "parsed": False},
                )
                return fallback

        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            raise

    async def generate_interview_question(
        self,
        resume_text: str,
        posting_text: str,
        interview_type: str = "technical",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate interview question

        Args:
            resume_text: Resume content
            posting_text: Job posting content
            interview_type: "technical" or "personality"

        Returns:
            Interview question as dict
        """
        try:
            if interview_type == "technical":
                prompt = f"""다음 이력서와 채용공고를 바탕으로 기술 면접 질문을 생성해주세요.

이력서:
{resume_text[:500]}...

채용공고:
{posting_text[:500]}...

JSON 형식으로 질문을 제공해주세요:
{{
  "question": "질문 내용",
  "difficulty": "easy|medium|hard",
  "category": "기술|인성|상황",
  "follow_up": false
}}"""
            else:
                prompt = f"""다음 이력서를 바탕으로 인성 면접 질문을 생성해주세요.

이력서:
{resume_text[:500]}...

JSON 형식으로 질문을 제공해주세요:
{{
  "question": "질문 내용",
  "difficulty": "easy|medium|hard",
  "category": "기술|인성|상황",
  "follow_up": false
}}"""

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            config = types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=512,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            # Extract JSON from response
            import json
            result_text = response.text

            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                self._langfuse_trace_and_generation(
                    trace_name="gemini_generate_interview_question",
                    generation_name="gemini_interview_question",
                    input_text=prompt,
                    output_text=result_text,
                    user_id=user_id,
                    metadata={"temperature": 0.8, "type": "interview_question", "interview_type": interview_type},
                )
                return parsed
            else:
                fallback = {
                    "question": result_text,
                    "difficulty": "medium",
                    "category": interview_type,
                    "follow_up": False
                }
                self._langfuse_trace_and_generation(
                    trace_name="gemini_generate_interview_question",
                    generation_name="gemini_interview_question",
                    input_text=prompt,
                    output_text=result_text,
                    user_id=user_id,
                    metadata={
                        "temperature": 0.8,
                        "type": "interview_question",
                        "interview_type": interview_type,
                        "parsed": False,
                    },
                )
                return fallback

        except Exception as e:
            logger.error(f"Error generating interview question: {e}")
            raise

    async def extract_text_from_file(
        self,
        file_url: str,
        file_type: str = "pdf",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from image or PDF file using Gemini Vision API
        
        Args:
            file_url: URL of the file (http(s):// or data:)
            file_type: "pdf" or "image"
            
        Returns:
            Dict with extracted_text and pages list
        """
        try:
            # 파일 단위 trace 생성 (페이지/이미지 generation을 여기에 연결)
            ocr_trace = trace_llm_call(
                name="gemini_extract_text",
                user_id=user_id,
                metadata={
                    "model": self.model_name,
                    "type": "ocr",
                    "file_type": file_type,
                    "file_url_prefix": file_url[:100],
                },
            )

            # Download file
            logger.info(f"Downloading file from {file_url[:100]}...")
            file_bytes = await self._download_file(file_url)
            
            if file_type == "pdf":
                # Convert PDF to images
                logger.info("Converting PDF to images...")
                images = self._pdf_to_images(file_bytes)
                
                # Extract text from each page
                pages = []
                full_text = ""
                
                for page_num, image in enumerate(images, start=1):
                    logger.info(f"Extracting text from page {page_num}/{len(images)}...")
                    page_text = await self._extract_text_from_image(image, user_id=user_id, page_num=page_num)
                    pages.append({
                        "page": page_num,
                        "text": page_text
                    })
                    full_text += f"\n\n[Page {page_num}]\n{page_text}"

                    # 페이지 단위 generation 기록 (trace가 있을 때만)
                    if ocr_trace is not None:
                        create_generation(
                            trace=ocr_trace,
                            name=f"gemini_ocr_page_{page_num}",
                            model=self.model_name,
                            input_text=f"file_type=pdf page={page_num} file_url_prefix={file_url[:100]}",
                            output_text=page_text[:4000],
                            metadata={
                                "type": "ocr",
                                "file_type": "pdf",
                                "page": page_num,
                                "total_pages": len(images),
                            },
                        )
                
                result = {
                    "extracted_text": full_text.strip(),
                    "pages": pages
                }
                # 파일 요약 generation(전체 텍스트는 너무 길 수 있어 4,000자까지만 저장)
                if ocr_trace is not None:
                    create_generation(
                        trace=ocr_trace,
                        name="gemini_ocr_pdf_summary",
                        model=self.model_name,
                        input_text=f"file_type=pdf file_url_prefix={file_url[:100]}",
                        output_text=result["extracted_text"][:4000],
                        metadata={"type": "ocr", "file_type": "pdf", "pages": len(pages)},
                    )
                return result
            else:
                # Single image
                logger.info("Extracting text from image...")
                image = Image.open(io.BytesIO(file_bytes))
                text = await self._extract_text_from_image(image, user_id=user_id, page_num=1)
                
                result = {
                    "extracted_text": text,
                    "pages": [{"page": 1, "text": text}]
                }
                if ocr_trace is not None:
                    create_generation(
                        trace=ocr_trace,
                        name="gemini_ocr_image",
                        model=self.model_name,
                        input_text=f"file_type=image file_url_prefix={file_url[:100]}",
                        output_text=text[:4000],
                        metadata={"type": "ocr", "file_type": "image", "pages": 1},
                    )
                return result
                
        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise
    
    async def _download_file(self, file_url: str) -> bytes:
        """Download file from URL"""
        # Handle data: URL
        if file_url.startswith('data:'):
            try:
                import base64
                header, encoded = file_url.split(',', 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"Failed to decode data URL: {e}")
                raise ValueError("Invalid data URL format")
        
        # HTTP(S) URL
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=30.0)
            response.raise_for_status()
            return response.content
    
    def _pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF to images"""
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
    
    async def _extract_text_from_image(
        self,
        image: Image.Image,
        user_id: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> str:
        """Extract text from image using Gemini Vision API"""
        try:
            # Convert image to bytes
            buffered = io.BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            
            # Create prompt for OCR
            prompt = types.Part.from_text(
                text="""Extract all text from this image. 
                
Return the text exactly as it appears, preserving the layout and structure.
Include all visible text including names, addresses, phone numbers, emails, dates, and any other information.

Return ONLY the extracted text, without any additional commentary or formatting."""
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
            
            config = types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for accurate extraction
                max_output_tokens=2048,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            extracted_text = response.text.strip()
            logger.info(f"Extracted {len(extracted_text)} characters from image")

            # 이미지 단위 OCR 호출도 별도 기록 (bytes는 남기지 않고, 메타만 남김)
            self._langfuse_trace_and_generation(
                trace_name="gemini_extract_text_from_image",
                generation_name="gemini_ocr_image_call",
                input_text="Extract all text from image (bytes omitted)",
                output_text=extracted_text[:4000],
                user_id=user_id,
                metadata={"type": "ocr_image_call", "page": page_num},
            )
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            raise
