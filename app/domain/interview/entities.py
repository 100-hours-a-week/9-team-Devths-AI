"""
Interview Domain Entities.

Defines the data structures for interview state management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict


class InterviewType(str, Enum):
    """Type of interview."""

    TECH = "tech"
    BEHAVIOR = "behavior"


class InterviewPhase(str, Enum):
    """Phase of interview workflow."""

    INIT = "init"
    QUESTIONING = "questioning"
    FOLLOWUP = "followup"
    COMPLETED = "completed"


@dataclass
class InterviewQuestion:
    """Single interview question with state."""

    id: int
    category: str
    question: str
    is_completed: bool = False
    current_depth: int = 0
    max_depth: int = 3
    conversation: list[dict[str, str]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.

        Args:
            role: Role (interviewer or candidate).
            content: Message content.
        """
        self.conversation.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "is_completed": self.is_completed,
            "current_depth": self.current_depth,
            "max_depth": self.max_depth,
            "conversation": self.conversation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterviewQuestion":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=data["category"],
            question=data["question"],
            is_completed=data.get("is_completed", False),
            current_depth=data.get("current_depth", 0),
            max_depth=data.get("max_depth", 3),
            conversation=data.get("conversation", []),
        )


@dataclass
class InterviewSession:
    """Interview session state."""

    session_id: str
    user_id: str
    interview_type: InterviewType
    phase: InterviewPhase = InterviewPhase.INIT
    questions: list[InterviewQuestion] = field(default_factory=list)
    current_question_idx: int = 0
    resume_text: str = ""
    job_posting_text: str = ""
    portfolio_text: str = ""
    full_conversation: list[dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime | None = None

    @property
    def current_question(self) -> InterviewQuestion | None:
        """Get current question."""
        if 0 <= self.current_question_idx < len(self.questions):
            return self.questions[self.current_question_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if interview is complete."""
        return self.phase == InterviewPhase.COMPLETED

    @property
    def total_questions(self) -> int:
        """Get total number of questions."""
        return len(self.questions)

    @property
    def answered_questions(self) -> int:
        """Get number of answered questions."""
        return sum(1 for q in self.questions if q.is_completed)

    def add_conversation_message(self, role: str, content: str) -> None:
        """Add a message to full conversation history.

        Args:
            role: Role (interviewer or candidate).
            content: Message content.
        """
        self.full_conversation.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "interview_type": self.interview_type.value,
            "phase": self.phase.value,
            "questions": [q.to_dict() for q in self.questions],
            "current_question_idx": self.current_question_idx,
            "resume_text": self.resume_text,
            "job_posting_text": self.job_posting_text,
            "portfolio_text": self.portfolio_text,
            "full_conversation": self.full_conversation,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterviewSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            interview_type=InterviewType(data["interview_type"]),
            phase=InterviewPhase(data["phase"]),
            questions=[InterviewQuestion.from_dict(q) for q in data.get("questions", [])],
            current_question_idx=data.get("current_question_idx", 0),
            resume_text=data.get("resume_text", ""),
            job_posting_text=data.get("job_posting_text", ""),
            portfolio_text=data.get("portfolio_text", ""),
            full_conversation=data.get("full_conversation", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=(
                datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
            ),
        )


# TypedDict for LangGraph State
class InterviewState(TypedDict):
    """Interview state for LangGraph workflow."""

    # Session info
    session_id: str
    user_id: str
    interview_type: str  # "tech" | "behavior"

    # Questions (list of dicts)
    questions: list[dict[str, Any]]
    current_question_idx: int
    current_depth: int  # 0-3 for follow-up questions

    # Context documents
    resume_text: str
    job_posting_text: str
    portfolio_text: str

    # Conversation
    messages: list[dict[str, str]]

    # Current state
    phase: str  # "init" | "questioning" | "followup" | "completed"
    response: str
    user_answer: str

    # Evaluation (for report generation)
    evaluation: dict[str, Any] | None
