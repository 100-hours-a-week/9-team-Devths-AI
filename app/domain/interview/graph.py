"""
LangGraph Interview Workflow Implementation.

Implements the interview state machine using LangGraph.
"""

import json
import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.infrastructure.llm.langchain_wrapper import LangChainLLMGateway

from .entities import InterviewState

logger = logging.getLogger(__name__)


# Interview Prompts
TECH_INTERVIEW_INIT_PROMPT = """당신은 기술 면접관입니다. 지원자의 이력서와 채용공고를 바탕으로 5개의 기술 면접 질문을 생성해주세요.

이력서:
{resume_text}

채용공고:
{job_posting_text}

{portfolio_section}

다음 JSON 형식으로 5개의 질문을 생성해주세요:
{{
    "questions": [
        {{
            "id": 1,
            "category": "cs_fundamentals|programming|system_design|problem_solving|experience",
            "question": "질문 내용"
        }},
        ...
    ]
}}

질문 생성 시 고려사항:
1. 이력서의 기술 스택과 경험을 기반으로 질문
2. 채용공고의 요구사항에 맞는 질문
3. 기초부터 심화까지 난이도 조절
4. 실무 경험을 물어보는 질문 포함"""

TECH_FOLLOWUP_PROMPT = """당신은 기술 면접관입니다. 지원자의 답변을 듣고 꼬리질문을 생성해주세요.

원래 질문: {original_question}
지원자 답변: {user_answer}
현재 꼬리질문 깊이: {current_depth}/3

다음 중 하나를 선택하세요:
1. 답변이 충분히 깊이 있다면 "다음 질문으로 넘어갑니다"라고 말하세요.
2. 더 깊은 이해가 필요하다면 관련 꼬리질문을 생성하세요.

응답 형식:
- 꼬리질문을 할 경우: 바로 질문만 작성
- 다음 질문으로 넘어갈 경우: "NEXT_QUESTION" 태그를 포함"""

INTERVIEW_REPORT_PROMPT = """당신은 기술 면접 평가자입니다. 면접 내용을 분석하고 평가 리포트를 작성해주세요.

면접 대화 내용:
{conversation}

다음 항목을 포함한 평가 리포트를 작성해주세요:
1. 전체적인 평가 (점수: 1-10)
2. 강점 분석
3. 개선이 필요한 부분
4. 질문별 상세 피드백
5. 향후 학습 추천

리포트는 한국어로 작성하고, 구체적이고 건설적인 피드백을 제공해주세요."""


def create_interview_graph(llm_gateway: LangChainLLMGateway) -> StateGraph:
    """Create LangGraph workflow for interview.

    Args:
        llm_gateway: LangChain LLM Gateway instance.

    Returns:
        Compiled StateGraph for interview workflow.
    """
    workflow = StateGraph(InterviewState)

    # Define nodes
    async def generate_questions(state: InterviewState) -> dict[str, Any]:
        """Generate 5 interview questions based on resume/job posting."""
        logger.info("Generating interview questions...")

        portfolio_section = ""
        if state.get("portfolio_text"):
            portfolio_section = f"\n포트폴리오:\n{state['portfolio_text']}"

        prompt = TECH_INTERVIEW_INIT_PROMPT.format(
            resume_text=state.get("resume_text", ""),
            job_posting_text=state.get("job_posting_text", ""),
            portfolio_section=portfolio_section,
        )

        response = await llm_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # Parse JSON response
        questions = []
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                questions = data.get("questions", [])
        except Exception as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            # Fallback: create default questions
            questions = [
                {"id": i + 1, "category": "general", "question": f"질문 {i + 1}"} for i in range(5)
            ]

        # Add state fields to questions
        for q in questions:
            q["is_completed"] = False
            q["current_depth"] = 0
            q["max_depth"] = 3
            q["conversation"] = []

        logger.info(f"Generated {len(questions)} questions")

        return {
            "questions": questions,
            "current_question_idx": 0,
            "current_depth": 0,
            "phase": "questioning",
        }

    async def ask_question(state: InterviewState) -> dict[str, Any]:
        """Ask the current question."""
        questions = state.get("questions", [])
        idx = state.get("current_question_idx", 0)

        if idx >= len(questions):
            return {"phase": "completed", "response": "면접이 완료되었습니다."}

        question = questions[idx]
        q_num = idx + 1
        total = len(questions)

        header = f"[기술면접 {q_num}/{total}]"
        response = f"{header}\n\n{question['question']}"

        # Add to messages
        messages = state.get("messages", [])
        messages.append({"role": "interviewer", "content": response})

        return {
            "response": response,
            "messages": messages,
            "phase": "questioning",
        }

    async def evaluate_answer(state: InterviewState) -> dict[str, Any]:
        """Evaluate the candidate's answer and decide next step."""
        user_answer = state.get("user_answer", "")
        questions = state.get("questions", [])
        idx = state.get("current_question_idx", 0)
        current_depth = state.get("current_depth", 0)

        if idx >= len(questions):
            return {"phase": "completed"}

        question = questions[idx]

        # Add answer to messages
        messages = state.get("messages", [])
        messages.append({"role": "candidate", "content": user_answer})

        # Add to question conversation
        question["conversation"].append(
            {
                "role": "candidate",
                "content": user_answer,
            }
        )

        # Check if max depth reached
        if current_depth >= question.get("max_depth", 3):
            # Mark as completed and move to next
            question["is_completed"] = True
            return {
                "questions": questions,
                "messages": messages,
                "phase": "next_question",
            }

        # Generate follow-up decision
        prompt = TECH_FOLLOWUP_PROMPT.format(
            original_question=question["question"],
            user_answer=user_answer,
            current_depth=current_depth,
        )

        response = await llm_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # Check if should move to next question
        if "NEXT_QUESTION" in response:
            question["is_completed"] = True
            return {
                "questions": questions,
                "messages": messages,
                "phase": "next_question",
            }

        # Continue with follow-up
        return {
            "questions": questions,
            "messages": messages,
            "response": response,
            "phase": "followup",
            "current_depth": current_depth + 1,
        }

    async def generate_followup(state: InterviewState) -> dict[str, Any]:
        """Generate and present a follow-up question."""
        response = state.get("response", "")
        messages = state.get("messages", [])
        questions = state.get("questions", [])
        idx = state.get("current_question_idx", 0)

        # Add follow-up to messages
        messages.append({"role": "interviewer", "content": response})

        # Add to question conversation
        if idx < len(questions):
            questions[idx]["conversation"].append(
                {
                    "role": "interviewer",
                    "content": response,
                }
            )
            questions[idx]["current_depth"] = state.get("current_depth", 0)

        return {
            "messages": messages,
            "questions": questions,
            "phase": "questioning",
        }

    async def next_question(state: InterviewState) -> dict[str, Any]:
        """Move to the next question."""
        idx = state.get("current_question_idx", 0)
        questions = state.get("questions", [])

        new_idx = idx + 1

        if new_idx >= len(questions):
            return {
                "current_question_idx": new_idx,
                "current_depth": 0,
                "phase": "completed",
            }

        return {
            "current_question_idx": new_idx,
            "current_depth": 0,
            "phase": "questioning",
        }

    async def generate_report(state: InterviewState) -> dict[str, Any]:
        """Generate the final interview evaluation report."""
        logger.info("Generating interview report...")

        messages = state.get("messages", [])

        # Format conversation
        conversation_text = "\n".join(
            [
                f"{'면접관' if m['role'] == 'interviewer' else '지원자'}: {m['content']}"
                for m in messages
            ]
        )

        prompt = INTERVIEW_REPORT_PROMPT.format(conversation=conversation_text)

        response = await llm_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for evaluation
        )

        # Parse evaluation if JSON
        evaluation = None
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                evaluation = json.loads(response[start:end])
        except Exception:
            pass

        return {
            "response": response,
            "evaluation": evaluation,
            "phase": "completed",
        }

    # Define routing function
    def route_after_evaluate(
        state: InterviewState,
    ) -> Literal["generate_followup", "next_question", "generate_report"]:
        """Route based on phase after evaluation."""
        phase = state.get("phase", "")
        idx = state.get("current_question_idx", 0)
        questions = state.get("questions", [])

        if phase == "followup":
            return "generate_followup"
        elif phase == "next_question":
            if idx + 1 >= len(questions):
                return "generate_report"
            return "next_question"
        elif phase == "completed":
            return "generate_report"

        return "next_question"

    def route_after_next(
        state: InterviewState,
    ) -> Literal["ask_question", "generate_report"]:
        """Route after moving to next question."""
        phase = state.get("phase", "")
        if phase == "completed":
            return "generate_report"
        return "ask_question"

    # Add nodes to workflow
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("ask_question", ask_question)
    workflow.add_node("evaluate_answer", evaluate_answer)
    workflow.add_node("generate_followup", generate_followup)
    workflow.add_node("next_question", next_question)
    workflow.add_node("generate_report", generate_report)

    # Define edges
    workflow.set_entry_point("generate_questions")
    workflow.add_edge("generate_questions", "ask_question")
    workflow.add_edge("generate_followup", "ask_question")

    # Conditional edges
    workflow.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluate,
        {
            "generate_followup": "generate_followup",
            "next_question": "next_question",
            "generate_report": "generate_report",
        },
    )

    workflow.add_conditional_edges(
        "next_question",
        route_after_next,
        {
            "ask_question": "ask_question",
            "generate_report": "generate_report",
        },
    )

    # End at report
    workflow.add_edge("generate_report", END)

    return workflow.compile()


class InterviewWorkflow:
    """Interview workflow manager using LangGraph."""

    def __init__(
        self,
        llm_gateway: LangChainLLMGateway,
    ):
        """Initialize interview workflow.

        Args:
            llm_gateway: LangChain LLM Gateway instance.
        """
        self._llm_gateway = llm_gateway
        self._graph = create_interview_graph(llm_gateway)
        logger.info("InterviewWorkflow initialized")

    async def start_interview(
        self,
        session_id: str,
        user_id: str,
        interview_type: str,
        resume_text: str,
        job_posting_text: str,
        portfolio_text: str = "",
    ) -> InterviewState:
        """Start a new interview session.

        Args:
            session_id: Unique session identifier.
            user_id: User identifier.
            interview_type: Type of interview (tech/behavior).
            resume_text: Resume content.
            job_posting_text: Job posting content.
            portfolio_text: Portfolio content (optional).

        Returns:
            Initial interview state with generated questions.
        """
        initial_state: InterviewState = {
            "session_id": session_id,
            "user_id": user_id,
            "interview_type": interview_type,
            "questions": [],
            "current_question_idx": 0,
            "current_depth": 0,
            "resume_text": resume_text,
            "job_posting_text": job_posting_text,
            "portfolio_text": portfolio_text,
            "messages": [],
            "phase": "init",
            "response": "",
            "user_answer": "",
            "evaluation": None,
        }

        # Run workflow until first question is asked
        config = {"configurable": {"thread_id": session_id}}
        result = await self._graph.ainvoke(initial_state, config)

        return result

    async def process_answer(
        self,
        state: InterviewState,
        user_answer: str,
    ) -> InterviewState:
        """Process user's answer and get next response.

        Args:
            state: Current interview state.
            user_answer: User's answer to the question.

        Returns:
            Updated interview state.
        """
        # Update state with answer
        state["user_answer"] = user_answer
        state["phase"] = "questioning"

        # Run evaluation and get next step
        config = {"configurable": {"thread_id": state["session_id"]}}

        # Manually invoke evaluate_answer and follow the workflow
        result = await self._graph.ainvoke(state, config)

        return result

    async def get_report(self, state: InterviewState) -> str:
        """Generate final interview report.

        Args:
            state: Final interview state.

        Returns:
            Interview evaluation report.
        """
        if state.get("phase") != "completed":
            # Force completion
            state["phase"] = "completed"

        config = {"configurable": {"thread_id": state["session_id"]}}
        result = await self._graph.ainvoke(state, config)

        return result.get("response", "")
