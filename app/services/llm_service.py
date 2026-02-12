"""
LLM Service using Gemini Flash API

"""

import asyncio
import contextlib
import io
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pdf2image
from google import genai
from google.genai import types
from PIL import Image

from app.config.settings import get_settings
from app.services.cloudwatch_service import CloudWatchService
from app.utils.langfuse_client import create_generation, trace_llm_call

logger = logging.getLogger(__name__)


class LLMService:
    """LLM Service for chatbot using Gemini Flash"""

    def __init__(self, api_key: str | None = None):
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
        self.model_name = "gemini-3-flash-preview"  # Gemini 3 Flash Preview
        # 분석용 모델도 동일하게 사용 (gemini-3-pro는 존재하지 않음)
        self.analysis_model = "gemini-3-flash-preview"

        # 중앙화된 설정 로드
        self._settings = get_settings()

        logger.info(f"LLM Service initialized with model: {self.model_name}")

    def _langfuse_trace_and_generation(
        self,
        *,
        trace_name: str,
        generation_name: str,
        input_text: str,
        output_text: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
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

    def _record_token_usage(self, response: Any, model_name: str) -> None:
        """CloudWatch에 토큰 사용량 기록"""
        try:
            usage = None
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata

            if usage:
                cw = CloudWatchService.get_instance()
                dims = {"Model": model_name, "Type": "Input"}
                prompt_tokens = usage.prompt_token_count or 0
                if prompt_tokens > 0:
                    asyncio.create_task(
                        cw.put_metric("LLM_Token_Usage", prompt_tokens, "Count", dims)
                    )

                dims["Type"] = "Output"
                candidate_tokens = usage.candidates_token_count or 0
                if candidate_tokens > 0:
                    asyncio.create_task(
                        cw.put_metric("LLM_Token_Usage", candidate_tokens, "Count", dims)
                    )

                # Total은 합산해서 기록
                dims["Type"] = "Total"
                total_tokens = usage.total_token_count or (prompt_tokens + candidate_tokens)
                if total_tokens > 0:
                    asyncio.create_task(
                        cw.put_metric("LLM_Token_Usage", total_tokens, "Count", dims)
                    )

        except Exception as e:
            logger.warning(f"Failed to record token usage: {e}")

    def _record_error(self, error: Exception, model_name: str) -> None:
        """CloudWatch에 에러 기록"""
        try:
            cw = CloudWatchService.get_instance()
            error_type = type(error).__name__
            dims = {"Model": model_name, "ErrorType": error_type}
            asyncio.create_task(cw.put_metric("LLM_Error_Count", 1, "Count", dims))
        except Exception as e:
            # CloudWatch 메트릭 기록 실패 시 본 서비스 동작에는 영향을 주지 않되, 원인 파악을 위해 로깅만 수행
            logger.warning(f"Failed to record LLM error metric to CloudWatch: {e}")

    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        user_id: str | None = None,
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
                types.Content(role="user", parts=[types.Part.from_text(text=final_message)])
            ]

            # Create config (중앙화 설정 사용)
            config = types.GenerateContentConfig(
                temperature=self._settings.llm_temperature_chat,
                top_p=0.9,
                top_k=40,
                max_output_tokens=self._settings.llm_max_tokens_chat,
                system_instruction=system_prompt if system_prompt else None,
            )

            # Generate streaming response
            response = self.client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config
            )

            # Stream chunks
            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # 토큰 사용량 기록 (스트리밍 완료 후 response 객체에서 확인)
            self._record_token_usage(response, self.model_name)

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
            self._record_error(e, self.model_name)  # 에러 메트릭 기록

            if trace is not None:
                with contextlib.suppress(Exception):
                    trace["client"].create_event(
                        trace_context={"trace_id": trace["trace_id"]},
                        name="error",
                        level="ERROR",
                        metadata={"error": str(e)},
                    )
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def generate_response_non_stream(
        self,
        user_message: str,
        context: str | None = None,
        system_prompt: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Generate non-streaming response from LLM (JSON 응답 등에 적합)

        Args:
            user_message: User's message
            context: RAG context from VectorDB (optional)
            system_prompt: System instructions (optional)
            user_id: User ID for tracing

        Returns:
            Complete response text
        """
        trace = trace_llm_call(
            name="gemini_generate_response_non_stream",
            user_id=user_id,
            metadata={
                "model": self.model_name,
                "has_context": bool(context),
                "streaming": False,
            },
        )

        try:
            # Build final user message with context
            final_message = user_message
            if context:
                final_message = f"""관련 정보:
{context}

질문: {user_message}

위 관련 정보를 참고하여 질문에 답변해주세요."""

            # Create contents using types.Content
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=final_message)])
            ]

            # Create config (중앙화 설정 사용)
            config = types.GenerateContentConfig(
                temperature=self._settings.llm_temperature_chat,
                top_p=0.9,
                top_k=40,
                max_output_tokens=self._settings.llm_max_tokens_chat,
                system_instruction=system_prompt if system_prompt else None,
            )

            # Generate non-streaming response
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )

            self._record_token_usage(response, self.model_name)  # 토큰 사용량 기록

            result_text = response.text if hasattr(response, "text") else ""

            # Langfuse generation 기록
            if trace is not None:
                create_generation(
                    trace=trace,
                    name="gemini_non_stream",
                    model=self.model_name,
                    input_text=final_message,
                    output_text=result_text,
                    metadata={"streaming": False},
                )

            return result_text

        except Exception as e:
            logger.error(f"Error generating LLM response (non-stream): {e}")
            self._record_error(e, self.model_name)  # 에러 메트릭 기록

            if trace is not None:
                with contextlib.suppress(Exception):
                    trace["client"].create_event(
                        trace_context={"trace_id": trace["trace_id"]},
                        name="error",
                        level="ERROR",
                        metadata={"error": str(e)},
                    )
            raise

    async def generate_analysis(
        self,
        resume_text: str,
        posting_text: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate resume and job posting analysis (단계별 분할 호출)

        Args:
            resume_text: Resume content
            posting_text: Job posting content

        Returns:
            Analysis result as dict
        """
        import json

        # 텍스트 길이 제한 (너무 길면 Gemini가 응답 못함)
        max_text_len = 4000
        resume_text_trimmed = (
            resume_text[:max_text_len] if len(resume_text) > max_text_len else resume_text
        )
        posting_text_trimmed = (
            posting_text[:max_text_len] if len(posting_text) > max_text_len else posting_text
        )

        logger.info(
            f"[분석] 이력서 길이: {len(resume_text_trimmed)}자, 채용공고 길이: {len(posting_text_trimmed)}자"
        )

        # 기본 설정 (중앙화 설정 사용)
        config = types.GenerateContentConfig(
            temperature=self._settings.llm_temperature_analysis,
            max_output_tokens=self._settings.llm_max_tokens_analysis,
        )

        # 결과 저장용
        final_result = {
            "resume_analysis": {"strengths": [], "weaknesses": [], "suggestions": []},
            "posting_analysis": {
                "company": "알 수 없음",
                "position": "알 수 없음",
                "required_skills": [],
                "preferred_skills": [],
            },
            "matching": {
                "score": 0,
                "grade": "F",
                "matched_skills": [],
                "missing_skills": [],
            },
        }

        # ========== 1단계: 이력서 분석 ==========
        try:
            resume_prompt = f"""당신은 취업 컨설턴트입니다. 아래 이력서를 분석해주세요.

[이력서]
{resume_text_trimmed}

아래 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요:
{{"strengths": ["강점1", "강점2", "강점3"], "weaknesses": ["약점1", "약점2"], "suggestions": ["제안1", "제안2"]}}"""

            resume_result = await self._call_gemini_with_retry(
                prompt=resume_prompt,
                config=config,
                step_name="이력서 분석",
            )

            if resume_result:
                parsed = self._parse_json_response(resume_result)
                if parsed:
                    final_result["resume_analysis"] = {
                        "strengths": parsed.get("strengths", []),
                        "weaknesses": parsed.get("weaknesses", []),
                        "suggestions": parsed.get("suggestions", []),
                    }
                    logger.info("[분석] ✅ 이력서 분석 완료")
        except Exception as e:
            logger.warning(f"[분석] 이력서 분석 실패: {e}")

        # ========== 2단계: 채용공고 분석 ==========
        try:
            posting_prompt = f"""당신은 채용 전문가입니다. 아래 채용공고를 분석해주세요.

[채용공고]
{posting_text_trimmed}

아래 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요:
{{"company": "회사명", "position": "직무/포지션", "required_skills": ["필수역량1", "필수역량2"], "preferred_skills": ["우대사항1", "우대사항2"]}}"""

            posting_result = await self._call_gemini_with_retry(
                prompt=posting_prompt,
                config=config,
                step_name="채용공고 분석",
            )

            if posting_result:
                parsed = self._parse_json_response(posting_result)
                if parsed:
                    final_result["posting_analysis"] = {
                        "company": parsed.get("company", "알 수 없음"),
                        "position": parsed.get("position", "알 수 없음"),
                        "required_skills": parsed.get("required_skills", []),
                        "preferred_skills": parsed.get("preferred_skills", []),
                    }
                    logger.info("[분석] ✅ 채용공고 분석 완료")
        except Exception as e:
            logger.warning(f"[분석] 채용공고 분석 실패: {e}")

        # ========== 3단계: 매칭도 분석 ==========
        try:
            # 이력서와 채용공고 분석이 성공한 경우에만 매칭도 분석
            resume_strengths = final_result["resume_analysis"].get("strengths", [])
            required_skills = final_result["posting_analysis"].get("required_skills", [])

            if resume_strengths and required_skills:
                matching_prompt = f"""당신은 취업 매칭 전문가입니다. 이력서 강점과 채용공고 요구사항을 비교해주세요.

[이력서 강점]
{', '.join(resume_strengths[:5])}

[이력서 약점]
{', '.join(final_result["resume_analysis"].get("weaknesses", [])[:5])}

[채용공고 필수 역량]
{', '.join(required_skills[:5])}

[채용공고 우대 사항]
{', '.join(final_result["posting_analysis"].get("preferred_skills", [])[:5])}

매칭도를 0~100점으로 평가하고, 아래 JSON 형식으로만 응답하세요:
{{"score": 75, "grade": "B", "matched_skills": ["매칭되는역량1", "매칭되는역량2"], "missing_skills": ["부족한역량1", "부족한역량2"]}}"""

                matching_result = await self._call_gemini_with_retry(
                    prompt=matching_prompt,
                    config=config,
                    step_name="매칭도 분석",
                )

                if matching_result:
                    parsed = self._parse_json_response(matching_result)
                    if parsed:
                        final_result["matching"] = {
                            "score": parsed.get("score", 0),
                            "grade": parsed.get("grade", "F"),
                            "matched_skills": parsed.get("matched_skills", []),
                            "missing_skills": parsed.get("missing_skills", []),
                        }
                        logger.info("[분석] ✅ 매칭도 분석 완료")
        except Exception as e:
            logger.warning(f"[분석] 매칭도 분석 실패: {e}")

        # Langfuse 기록
        self._langfuse_trace_and_generation(
            trace_name="gemini_generate_analysis",
            generation_name="gemini_analysis_multi_step",
            input_text=f"이력서: {len(resume_text_trimmed)}자, 채용공고: {len(posting_text_trimmed)}자",
            output_text=json.dumps(final_result, ensure_ascii=False),
            user_id=user_id,
            metadata={"temperature": 0.3, "type": "analysis", "multi_step": True},
        )

        return final_result

    async def _call_gemini_with_retry(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        step_name: str,
        max_retries: int = 3,
    ) -> str | None:
        """Gemini API 호출 (재시도 로직 포함)"""
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.analysis_model, contents=contents, config=config
                )

                # 응답 텍스트 추출
                result_text = None

                # candidates에서 추출 시도
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]

                    # finish_reason 확인
                    if hasattr(candidate, "finish_reason"):
                        logger.debug(f"[{step_name}] finish_reason: {candidate.finish_reason}")

                    # content.parts에서 텍스트 추출
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and hasattr(candidate.content, "parts")
                        and candidate.content.parts
                    ):
                        result_text = candidate.content.parts[0].text

                # response.text로 fallback
                if not result_text:
                    with contextlib.suppress(Exception):
                        result_text = response.text

                if result_text:
                    logger.info(
                        f"[{step_name}] 응답 성공 (시도 {attempt + 1}/{max_retries}, {len(result_text)}자)"
                    )
                    self._record_token_usage(response, self.analysis_model)  # 토큰 사용량 기록
                    return result_text

                # 빈 응답 - 프롬프트 피드백 확인
                if hasattr(response, "prompt_feedback"):
                    logger.warning(f"[{step_name}] prompt_feedback: {response.prompt_feedback}")

                logger.warning(f"[{step_name}] 시도 {attempt + 1}/{max_retries} 실패 - 빈 응답")

            except Exception as e:
                logger.warning(f"[{step_name}] 시도 {attempt + 1}/{max_retries} 예외: {e}")
                self._record_error(e, self.analysis_model)  # 에러 메트릭 기록

        logger.error(f"[{step_name}] {max_retries}회 재시도 후에도 실패")
        return None

    def _parse_json_response(self, text: str) -> dict | None:
        """JSON 응답 파싱 (코드블록 처리 포함)"""
        import json

        if not text:
            return None

        try:
            # 먼저 바로 파싱 시도
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 코드 블록 제거
        clean_text = text.strip()
        if clean_text.startswith("```"):
            # ```json 또는 ``` 제거
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        # JSON 추출
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = clean_text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 파싱 실패: {e}")
                return None

        return None

    async def generate_interview_question(
        self,
        resume_text: str,
        posting_text: str,
        interview_type: str = "technical",
        user_id: str | None = None,
        previous_feedback: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate interview question

        Args:
            resume_text: Resume content
            posting_text: Job posting content
            interview_type: "technical" or "personality"
            previous_feedback: Optional previous interview feedback (RAG from interview_feedback)

        Returns:
            Interview question as dict
        """
        try:
            feedback_block = ""
            if previous_feedback:
                feedback_block = f"""
참고 - 이전 면접에서의 약점/피드백:
{previous_feedback[:800]}...
"""
            if interview_type == "technical":
                prompt = f"""다음 이력서와 채용공고를 바탕으로 기술 면접 질문을 생성해주세요.

이력서:
{resume_text[:500]}...

채용공고:
{posting_text[:500]}...
{feedback_block}
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
{feedback_block}
JSON 형식으로 질문을 제공해주세요:
{{
  "question": "질문 내용",
  "difficulty": "easy|medium|hard",
  "category": "기술|인성|상황",
  "follow_up": false
}}"""

            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

            config = types.GenerateContentConfig(
                temperature=self._settings.llm_temperature_interview_question,
                max_output_tokens=self._settings.llm_max_tokens_interview,
            )

            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )

            # Extract JSON from response
            import json

            result_text = response.text

            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                parsed = json.loads(json_str)
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
                    },
                )
                return parsed
            else:
                fallback = {
                    "question": result_text,
                    "difficulty": "medium",
                    "category": interview_type,
                    "follow_up": False,
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

    async def generate_interview_questions_batch(
        self,
        resume_text: str,
        posting_text: str,
        interview_type: str = "technical",
        count: int = 5,
        user_id: str | None = None,
        previous_feedback: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        한번에 여러 면접 질문을 구조화하여 생성

        Args:
            resume_text: Resume content
            posting_text: Job posting content
            interview_type: "technical" or "personality"
            count: Number of questions to generate (default: 5)
            user_id: User ID for tracing
            previous_feedback: Optional previous interview feedback

        Returns:
            List of interview questions as dicts
        """
        import json

        try:
            feedback_block = ""
            if previous_feedback:
                feedback_block = f"""
참고 - 이전 면접에서의 약점/피드백:
{previous_feedback[:800]}
"""

            type_label = "기술 면접" if interview_type == "technical" else "인성 면접"

            prompt = f"""다음 이력서와 채용공고를 바탕으로 {type_label} 질문을 {count}개 생성해주세요.

이력서:
{resume_text[:800]}

채용공고:
{posting_text[:800]}
{feedback_block}
질문들은 서로 다른 주제와 난이도를 가져야 합니다.
난이도는 easy, medium, hard를 적절히 섞어주세요.

반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요:
[
  {{"question": "질문 내용 1", "difficulty": "easy", "category": "{type_label}", "follow_up": false}},
  {{"question": "질문 내용 2", "difficulty": "medium", "category": "{type_label}", "follow_up": false}},
  {{"question": "질문 내용 3", "difficulty": "medium", "category": "{type_label}", "follow_up": false}},
  {{"question": "질문 내용 4", "difficulty": "hard", "category": "{type_label}", "follow_up": false}},
  {{"question": "질문 내용 5", "difficulty": "hard", "category": "{type_label}", "follow_up": false}}
]"""

            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

            config = types.GenerateContentConfig(
                temperature=self._settings.llm_temperature_interview_question,
                max_output_tokens=self._settings.llm_max_tokens_interview * 2,  # 배치이므로 2배
            )

            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )

            self._record_token_usage(response, self.model_name)
            result_text = response.text if hasattr(response, "text") else ""

            # JSON 배열 파싱
            clean_text = result_text.strip()

            # 코드 블록 제거
            if clean_text.startswith("```"):
                clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            # JSON 배열 추출
            start_idx = clean_text.find("[")
            end_idx = clean_text.rfind("]") + 1

            questions = []
            if start_idx != -1 and end_idx > start_idx:
                json_str = clean_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    questions = parsed[:count]

            # Langfuse 기록
            self._langfuse_trace_and_generation(
                trace_name="gemini_generate_interview_questions_batch",
                generation_name="gemini_interview_batch",
                input_text=prompt[:2000],
                output_text=result_text[:4000],
                user_id=user_id,
                metadata={
                    "temperature": self._settings.llm_temperature_interview_question,
                    "type": "interview_question_batch",
                    "interview_type": interview_type,
                    "requested_count": count,
                    "actual_count": len(questions),
                },
            )

            if questions:
                logger.info(
                    f"✅ 면접 질문 배치 생성 완료: {len(questions)}개 ({interview_type})"
                )
                return questions

            # 배치 파싱 실패 시 개별 생성으로 폴백
            logger.warning("배치 파싱 실패, 개별 생성으로 폴백")
            fallback_questions = []
            for _ in range(count):
                q = await self.generate_interview_question(
                    resume_text=resume_text,
                    posting_text=posting_text,
                    interview_type=interview_type,
                    user_id=user_id,
                    previous_feedback=previous_feedback,
                )
                fallback_questions.append(q)
            return fallback_questions

        except Exception as e:
            logger.error(f"Error generating batch interview questions: {e}")
            raise

    async def extract_text_from_file(
        self,
        file_url: str,
        file_type: str = "pdf",
        user_id: str | None = None,
    ) -> dict[str, Any]:
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
                    page_text = await self._extract_text_from_image(
                        image, user_id=user_id, page_num=page_num
                    )
                    pages.append({"page": page_num, "text": page_text})
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

                result = {"extracted_text": full_text.strip(), "pages": pages}
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

                result = {"extracted_text": text, "pages": [{"page": 1, "text": text}]}
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
            logger.info(f"Converted PDF to {len(images)} images")
            return images
        finally:
            os.unlink(tmp_path)

    async def _extract_text_from_image(
        self,
        image: Image.Image,
        user_id: str | None = None,
        page_num: int | None = None,
    ) -> str:
        """Extract text from image using Gemini Vision API"""
        try:
            # Convert image to bytes
            buffered = io.BytesIO()
            if image.mode != "RGB":
                image = image.convert("RGB")
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
                    parts=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), prompt],
                )
            ]

            config = types.GenerateContentConfig(
                temperature=self._settings.llm_temperature_ocr,
                max_output_tokens=self._settings.llm_max_tokens_chat,
            )

            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
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
