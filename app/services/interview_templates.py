"""
면접 질문 템플릿 서비스.

이력서/채용공고에서 기술 키워드를 추출하고,
템플릿 기반으로 면접 질문을 즉시 생성합니다 (LLM 호출 없이).

꼬리질문만 LLM이 맥락에 맞게 생성합니다.
"""

import logging
import random
import re

logger = logging.getLogger(__name__)

# ============================================================
# 기술 키워드 사전 (이력서/채용공고에서 매칭)
# ============================================================

TECH_KEYWORDS = [
    # 프로그래밍 언어
    "Java",
    "Python",
    "JavaScript",
    "TypeScript",
    "Kotlin",
    "Go",
    "Rust",
    "C\\+\\+",
    "C#",
    "Swift",
    "Ruby",
    "PHP",
    "Scala",
    # 프론트엔드
    "React",
    "Vue",
    "Angular",
    "Next\\.js",
    "Nuxt",
    "Svelte",
    "HTML",
    "CSS",
    "Tailwind",
    # 백엔드
    "Spring Boot",
    "Spring",
    "Django",
    "Flask",
    "FastAPI",
    "Express",
    "NestJS",
    "Node\\.js",
    "Rails",
    # 데이터베이스
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "DynamoDB",
    "Oracle",
    "MariaDB",
    "SQLite",
    # 인프라/DevOps
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "Terraform",
    "Jenkins",
    "GitHub Actions",
    "CI/CD",
    "Nginx",
    # AI/ML
    "PyTorch",
    "TensorFlow",
    "LangChain",
    "Hugging Face",
    "OpenAI",
    "RAG",
    "LLM",
    # 기타
    "Git",
    "Linux",
    "Kafka",
    "RabbitMQ",
    "GraphQL",
    "gRPC",
    "REST API",
    "WebSocket",
    "Figma",
    "Jira",
]

# ============================================================
# 기술 면접 질문 템플릿 (난이도별)
# ============================================================

TECHNICAL_TEMPLATES = {
    "easy": [
        "{tech}을(를) 프로젝트에서 어떻게 활용하셨나요?",
        "{tech}을(를) 사용하게 된 계기는 무엇인가요?",
        "{tech}의 주요 특징이나 장점은 무엇이라고 생각하시나요?",
        "{tech}을(를) 배우면서 가장 인상 깊었던 점은 무엇인가요?",
    ],
    "medium": [
        "{tech}을(를) 선택한 이유와 고려한 대안은 무엇이었나요?",
        "{tech} 사용 시 겪은 문제와 해결 방법을 설명해주세요.",
        "{tech}의 성능을 개선하기 위해 시도한 방법이 있나요?",
        "{tech}을(를) 활용한 프로젝트의 아키텍처를 설명해주세요.",
        "{tech}에서 테스트는 어떻게 진행하셨나요?",
    ],
    "hard": [
        "{tech}에서 겪은 가장 어려운 기술적 문제와 해결 과정을 상세히 설명해주세요.",
        "{tech}의 내부 동작 원리를 설명해주실 수 있나요?",
        "{tech}을(를) 대규모 트래픽 환경에서 사용한다면 어떤 점을 고려해야 할까요?",
        "{tech}의 한계점은 무엇이고, 어떻게 보완할 수 있을까요?",
    ],
}

# ============================================================
# 인성 면접 질문 풀 (고정, 난이도별)
# ============================================================

PERSONALITY_QUESTIONS = {
    "easy": [
        "간단하게 자기소개 부탁드립니다.",
        "우리 회사에 지원하신 이유가 무엇인가요?",
        "이 직무를 선택하게 된 계기가 무엇인가요?",
        "본인의 강점과 약점을 말씀해 주세요.",
    ],
    "medium": [
        "팀 프로젝트에서 갈등이 생겼을 때 어떻게 해결하셨나요?",
        "동료와 의견 충돌이 있을 때 어떻게 대처하시나요?",
        "스트레스를 받을 때 어떻게 해소하시나요?",
        "협업할 때 본인이 주로 맡는 역할은 무엇인가요?",
        "리더십을 발휘한 경험이 있다면 말씀해주세요.",
    ],
    "hard": [
        "가장 어려웠던 상황과 그것을 극복한 경험을 말씀해 주세요.",
        "실패한 경험이 있다면, 그것에서 무엇을 배웠나요?",
        "5년 후 본인의 모습을 어떻게 그리고 계신가요?",
        "인생에서 가장 중요하게 생각하는 가치는 무엇인가요?",
    ],
}


class InterviewTemplateService:
    """템플릿 기반 면접 질문 생성 서비스.

    - 기술 면접: 이력서/채용공고에서 키워드 추출 → 템플릿 대입
    - 인성 면접: 고정 질문 풀에서 랜덤 선택
    - LLM 호출 없이 즉시 생성 (0.01초 이내)
    """

    def __init__(self):
        self._rng = random.Random()
        # 키워드 매칭용 정규식 사전 컴파일
        self._keyword_patterns = [
            (re.compile(rf"\b{kw}\b", re.IGNORECASE), kw.replace("\\", "")) for kw in TECH_KEYWORDS
        ]

    def extract_tech_keywords(
        self,
        resume_text: str,
        posting_text: str | None = None,
    ) -> list[str]:
        """이력서/채용공고에서 기술 키워드를 추출합니다.

        Args:
            resume_text: 이력서 텍스트
            posting_text: 채용공고 텍스트 (선택)

        Returns:
            추출된 기술 키워드 리스트 (중복 제거)
        """
        combined = resume_text or ""
        if posting_text:
            combined += "\n" + posting_text

        found = []
        for pattern, display_name in self._keyword_patterns:
            if pattern.search(combined):
                found.append(display_name)

        logger.info(f"기술 키워드 추출: {len(found)}개 → {found[:10]}")
        return found

    def generate_questions(
        self,
        resume_text: str,
        posting_text: str = "",
        interview_type: str = "technical",
        count: int = 5,
        asked_questions: list[str] | None = None,
    ) -> list[dict]:
        """템플릿 기반 면접 질문 생성.

        Args:
            resume_text: 이력서 텍스트
            posting_text: 채용공고 텍스트
            interview_type: "technical" 또는 "personality"
            count: 생성할 질문 수
            asked_questions: 이미 질문한 목록 (중복 방지)

        Returns:
            [{"question": str, "difficulty": str, "category": str, "follow_up": False}, ...]
        """
        asked = set(asked_questions or [])

        if interview_type == "technical":
            return self._generate_technical(resume_text, posting_text, count, asked)
        return self._generate_personality(count, asked)

    def _generate_technical(
        self,
        resume_text: str,
        posting_text: str,
        count: int,
        asked: set[str],
    ) -> list[dict]:
        """기술 면접 질문 생성 (키워드 + 템플릿)."""
        keywords = self.extract_tech_keywords(resume_text, posting_text)

        if not keywords:
            logger.warning("기술 키워드 없음 → 범용 기술 질문 사용")
            keywords = ["기술 스택", "프로젝트", "개발"]

        # 난이도 배분: easy 1 + medium 2 + hard 2 (5개 기준)
        difficulty_plan = self._difficulty_plan(count)
        questions = []

        self._rng.shuffle(keywords)

        for difficulty in difficulty_plan:
            templates = TECHNICAL_TEMPLATES.get(difficulty, TECHNICAL_TEMPLATES["medium"])
            # 키워드 순환
            tech = keywords[len(questions) % len(keywords)]

            # 사용 안 한 템플릿에서 선택
            candidates = [
                t.format(tech=tech)
                for t in templates
                if t.format(tech=tech) not in asked
                and t.format(tech=tech) not in {q["question"] for q in questions}
            ]

            question_text = (
                self._rng.choice(candidates)
                if candidates
                else self._rng.choice(templates).format(tech=tech)
            )

            questions.append(
                {
                    "question": question_text,
                    "difficulty": difficulty,
                    "category": "기술 면접",
                    "follow_up": False,
                }
            )

        logger.info(f"템플릿 기술 질문 {len(questions)}개 생성 완료")
        return questions[:count]

    def _generate_personality(
        self,
        count: int,
        asked: set[str],
    ) -> list[dict]:
        """인성 면접 질문 생성 (고정 풀에서 랜덤 선택)."""
        difficulty_plan = self._difficulty_plan(count)
        questions = []

        for difficulty in difficulty_plan:
            pool = PERSONALITY_QUESTIONS.get(difficulty, PERSONALITY_QUESTIONS["medium"])
            candidates = [
                q for q in pool if q not in asked and q not in {q_["question"] for q_ in questions}
            ]

            question_text = self._rng.choice(candidates) if candidates else self._rng.choice(pool)

            questions.append(
                {
                    "question": question_text,
                    "difficulty": difficulty,
                    "category": "인성 면접",
                    "follow_up": False,
                }
            )

        logger.info(f"템플릿 인성 질문 {len(questions)}개 생성 완료")
        return questions[:count]

    def _difficulty_plan(self, count: int) -> list[str]:
        """난이도 배분 계획 (easy → medium → hard)."""
        if count <= 2:
            return ["easy", "medium"][:count]
        if count <= 3:
            return ["easy", "medium", "hard"]
        if count <= 5:
            plan = ["easy"] + ["medium"] * 2 + ["hard"] * 2
            return plan[:count]
        # 6개 이상
        plan = ["easy"] * max(1, count // 5)
        plan += ["medium"] * max(2, count * 2 // 5)
        plan += ["hard"] * max(2, count * 2 // 5)
        return plan[:count]
