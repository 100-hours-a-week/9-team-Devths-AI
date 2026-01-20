"""
vLLM Service for Llama-3-Korean-Bllossom-8B

Provides chat functionality and OCR using vLLM API server.
"""

import os
import io
import logging
import httpx
import tempfile
from typing import List, Dict, Any, AsyncIterator, Optional
from PIL import Image
import pdf2image

logger = logging.getLogger(__name__)


class VLLMService:
    """vLLM Service for chatbot using GCP GPU server"""

    def __init__(self, base_url: Optional[str] = None, ocr_only: bool = False, model_name: Optional[str] = None):
        """
        Initialize vLLM Service

        Args:
            base_url: GCP vLLM server URL (default: from GCP_VLLM_BASE_URL env)
            ocr_only: If True, only OCR functionality is available
            model_name: Model name (default: from VLLM_MODEL_NAME env)
        """
        self.ocr_only = ocr_only

        if ocr_only:
            logger.info("🔧 vLLM Service initialized in OCR-only mode (pytesseract)")
            logger.info("💰 가성비 모드: vLLM 서버 없이 pytesseract OCR만 사용")
            self.base_url = None
            self.model_name = "pytesseract-ocr"
            return

        # GCP vLLM 서버 URL 읽기
        gcp_url = os.getenv("GCP_VLLM_BASE_URL")
        self.base_url = base_url or gcp_url

        if not self.base_url:
            logger.warning("⚠️ GCP_VLLM_BASE_URL이 설정되지 않았습니다. OCR 전용 모드로 전환합니다.")
            self.ocr_only = True
            self.model_name = "pytesseract-ocr"
            return

        # vLLM 모델명 읽기
        self.model_name = model_name or os.getenv("VLLM_MODEL_NAME", "MLP-KTLim/llama-3-Korean-Bllossom-8B")

        # 로깅
        logger.info(f"🌐 vLLM Service initialized with GCP GPU server")
        logger.info(f"📍 GCP URL: {self.base_url}")
        logger.info(f"📦 Model: {self.model_name}")

    def get_server_info(self) -> Dict[str, Any]:
        """Get GCP vLLM server information"""
        return {
            "server_url": self.base_url,
            "model_name": self.model_name,
            "ocr_only": self.ocr_only
        }

    async def generate_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
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
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add context if provided
            if context:
                messages.append({
                    "role": "system",
                    "content": f"관련 정보:\n{context}\n\n위 관련 정보를 참고하여 질문에 답변해주세요."
                })
            
            # Add history
            if history:
                for msg in history:
                    # Handle both dict and Pydantic model
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                    else:
                        # Pydantic model (ChatMessage)
                        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                        content = msg.content
                    
                    messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })

            # Prepare request
            payload = {
                "model": self.model_name,  # GCP에 떠 있는 모델명
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
                "stream": True
            }

            # Make streaming request
            # URL 검증 및 정규화
            api_url = f"{self.base_url}/v1/chat/completions"
            
            # URL이 유효한지 확인
            if not api_url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL format: {api_url}. URL must start with http:// or https://")
            
            # Colab ngrok URL은 HTTPS일 수 있으므로 verify=False 허용 (개발 환경)
            verify_ssl = os.getenv("VLLM_VERIFY_SSL", "true").lower() == "true"
            
            logger.debug(f"vLLM API 요청 URL: {api_url}")
            
            async with httpx.AsyncClient(timeout=60.0, verify=verify_ssl) as client:
                try:
                    async with client.stream(
                        "POST",
                        api_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
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
                                                logger.info(f"vLLM 청크 {chunk_count}: {content[:50]}...")
                                            yield content
                                except json.JSONDecodeError as e:
                                    logger.debug(f"JSON 파싱 스킵: {data_str[:50]}")
                                    continue
                                except Exception as e:
                                    logger.warning(f"Error parsing vLLM response chunk: {e}, data: {data_str[:100]}")
                                    continue
                            elif line.strip():  # 빈 라인이 아닌 경우 로깅
                                logger.debug(f"vLLM 응답 라인 (data: 없음): {line[:100]}")
                        
                        if chunk_count == 0:
                            logger.warning("vLLM 응답에서 청크를 받지 못했습니다. 응답 형식을 확인하세요.")
                                
                except httpx.ConnectError as e:
                    logger.error(f"vLLM 서버 연결 실패: {self.base_url}")
                    logger.error(f"연결 오류: {str(e)}")
                    raise Exception(f"vLLM 서버에 연결할 수 없습니다. URL을 확인하세요: {self.base_url}")
                except httpx.TimeoutException:
                    logger.error(f"vLLM 서버 응답 시간 초과: {self.base_url}")
                    raise Exception("vLLM 서버 응답 시간이 초과되었습니다.")
                except httpx.HTTPStatusError as e:
                    logger.error(f"vLLM HTTP 오류: {e.response.status_code} - {e.response.text}")
                    raise Exception(f"vLLM 서버 오류: {e.response.status_code}")

        except Exception as e:
            logger.error(f"Error generating vLLM response: {e}")
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def extract_text_from_file(
        self,
        file_url: str,
        file_type: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Extract text from image or PDF file using pytesseract OCR
        
        Args:
            file_url: URL of the file (http(s):// or data:)
            file_type: "pdf" or "image"
            
        Returns:
            Dict with extracted_text and pages list
        """
        try:
            import pytesseract
            
            # Download file
            logger.info(f"[vLLM OCR] Downloading file from {file_url[:100]}...")
            file_bytes = await self._download_file(file_url)
            
            if file_type == "pdf":
                # Convert PDF to images
                logger.info("[vLLM OCR] Converting PDF to images...")
                images = self._pdf_to_images(file_bytes)
                
                # Extract text from each page using pytesseract
                pages = []
                full_text = ""
                
                for page_num, image in enumerate(images, start=1):
                    logger.info(f"[vLLM OCR] Extracting text from page {page_num}/{len(images)} using pytesseract...")
                    page_text = pytesseract.image_to_string(image, lang='kor+eng')
                    pages.append({
                        "page": page_num,
                        "text": page_text
                    })
                    full_text += f"\n\n[Page {page_num}]\n{page_text}"
                
                logger.info(f"[vLLM OCR] Extracted {len(full_text)} characters from {len(pages)} pages")
                return {
                    "extracted_text": full_text.strip(),
                    "pages": pages
                }
            else:
                # Single image
                logger.info("[vLLM OCR] Extracting text from image using pytesseract...")
                image = Image.open(io.BytesIO(file_bytes))
                text = pytesseract.image_to_string(image, lang='kor+eng')
                
                logger.info(f"[vLLM OCR] Extracted {len(text)} characters from image")
                return {
                    "extracted_text": text,
                    "pages": [{"page": 1, "text": text}]
                }
                
        except Exception as e:
            logger.error(f"[vLLM OCR] Error extracting text from file: {e}")
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
            logger.info(f"[vLLM OCR] Converted PDF to {len(images)} images")
            return images
        finally:
            os.unlink(tmp_path)
