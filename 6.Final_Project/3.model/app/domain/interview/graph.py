"""
LangGraph Interview Workflow Implementation.

Implements the interview state machine using LangGraph.
Uses prompt templates from app.prompts.interview (tech_interview_init, tech_followup, interview_report).
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, StateGraph

from app.infrastructure.llm.langchain_wrapper import LangChainLLMGateway
from app.prompts.interview import (
    create_interview_report_prompt,
    create_tech_followup_prompt,
    create_tech_interview_init_prompt,
    format_conversation_history,
)

from .entities import InterviewState

if TYPE_CHECKING:
    from app.infrastructure.session.base import BaseSessionStore

logger = logging.getLogger(__name__)

QUESTION_HISTORY_KEY_PREFIX = "question_history:"
QUESTION_HISTORY_TTL_SEC = 24 * 3600  # 24시간


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

        prompt = create_tech_interview_init_prompt(
            resume_text=state.get("resume_text", ""),
            job_posting_text=state.get("job_posting_text", ""),
            portfolio_text=state.get("portfolio_text", "") or "",
            previous_questions=state.get("previous_questions", []),
        )

        response = await llm_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        questions = []
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                raw = data.get("questions", [])
                for q in raw:
                    questions.append(
                        {
                            "id": q.get("id", len(questions) + 1),
                            "category": q.get("category", q.get("category_name", "general")),
                            "category_name": q.get("category_name", q.get("category", "일반")),
                            "question": q.get("question", ""),
                            "is_completed": False,
                            "current_depth": 0,
                            "max_depth": 3,
                            "conversation": [],
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            questions = [
                {
                    "id": i + 1,
                    "category": "general",
                    "category_name": "일반",
                    "question": f"질문 {i + 1}",
                    "is_completed": False,
                    "current_depth": 0,
                    "max_depth": 3,
                    "conversation": [],
                }
                for i in range(5)
            ]

        if len(questions) < 5:
            for i in range(len(questions), 5):
                questions.append(
                    {
                        "id": i + 1,
                        "category": "general",
                        "category_name": "일반",
                        "question": f"질문 {i + 1}",
                        "is_completed": False,
                        "current_depth": 0,
                        "max_depth": 3,
                        "conversation": [],
                    }
                )

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

        conversation_history = format_conversation_history(question.get("conversation", []))
        category_name = question.get("category_name") or question.get("category") or "일반"
        prompt = create_tech_followup_prompt(
            question_id=question.get("id", idx + 1),
            category_name=category_name,
            original_question=question["question"],
            conversation_history=conversation_history,
            last_answer=user_answer,
            current_depth=current_depth,
        )

        response = await llm_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        should_continue = True
        followup_question = response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                should_continue = data.get("should_continue", True)
                followup = data.get("followup")
                if followup and isinstance(followup, dict):
                    followup_question = followup.get("question", response)
        except Exception:
            pass

        if not should_continue:
            question["is_completed"] = True
            return {
                "questions": questions,
                "messages": messages,
                "phase": "next_question",
            }

        return {
            "questions": questions,
            "messages": messages,
            "response": followup_question,
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
        qa_history = "\n".join(
            [
                f"{'면접관' if m['role'] == 'interviewer' else '지원자'}: {m['content']}"
                for m in messages
            ]
        )

        prompt = create_interview_report_prompt(
            qa_history=qa_history,
            resume_text=state.get("resume_text", ""),
            job_posting_text=state.get("job_posting_text", ""),
        )

        response = await llm_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

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
        session_store: "BaseSessionStore | None" = None,
    ):
        """Initialize interview workflow.

        Args:
            llm_gateway: LangChain LLM Gateway instance.
            session_store: Optional session store for question history (중복 방지).
        """
        self._llm_gateway = llm_gateway
        self._graph = create_interview_graph(llm_gateway)
        self._session_store = session_store
        logger.info("InterviewWorkflow initialized")

    def _question_history_key(self, user_id: str) -> str:
        return f"{QUESTION_HISTORY_KEY_PREFIX}{user_id}"

    async def _load_previous_questions(self, user_id: str) -> list[str]:
        if not self._session_store:
            return []
        key = self._question_history_key(user_id)
        try:
            data = await self._session_store.get(key)
            if data and isinstance(data.get("questions"), list):
                return list(data["questions"])
        except Exception as e:
            logger.warning("Failed to load question history for %s: %s", user_id, e)
        return []

    async def _save_question_history(self, user_id: str, new_questions: list[str]) -> None:
        if not self._session_store or not new_questions:
            return
        key = self._question_history_key(user_id)
        try:
            existing = await self._session_store.get(key)
            questions = (
                list(existing["questions"]) if existing and existing.get("questions") else []
            )
            questions.extend(new_questions)
            await self._session_store.set(
                key,
                {"questions": questions},
                ttl=QUESTION_HISTORY_TTL_SEC,
            )
            logger.debug("Saved %d questions to history for user %s", len(new_questions), user_id)
        except Exception as e:
            logger.warning("Failed to save question history for %s: %s", user_id, e)

    async def start_interview(
        self,
        session_id: str,
        user_id: str,
        interview_type: str,
        resume_text: str,
        job_posting_text: str,
        portfolio_text: str = "",
        previous_questions: list[str] | None = None,
    ) -> InterviewState:
        """Start a new interview session.

        Args:
            session_id: Unique session identifier.
            user_id: User identifier.
            interview_type: Type of interview (tech/behavior).
            resume_text: Resume content.
            job_posting_text: Job posting content.
            portfolio_text: Portfolio content (optional).
            previous_questions: Optional override; if None and session_store set, loads from store.

        Returns:
            Initial interview state with generated questions.
        """
        if previous_questions is None and self._session_store:
            previous_questions = await self._load_previous_questions(user_id)
        previous_questions = previous_questions or []

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
            "previous_questions": previous_questions,
        }

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
        state["user_answer"] = user_answer
        state["phase"] = "questioning"

        config = {"configurable": {"thread_id": state["session_id"]}}
        result = await self._graph.ainvoke(state, config)

        if result.get("phase") == "completed":
            questions = result.get("questions") or []
            question_texts = [q.get("question", "").strip() for q in questions if q.get("question")]
            if question_texts and state.get("user_id"):
                await self._save_question_history(state["user_id"], question_texts)

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
