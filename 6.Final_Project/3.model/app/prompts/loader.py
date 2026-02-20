"""
YAML 기반 프롬프트 템플릿 로더.

역할별(system/human/ai) 키로 정의된 YAML 파일을 로드해
ChatPromptTemplate 등에서 사용할 수 있는 dict를 반환합니다.
"""

from pathlib import Path

import yaml

TEMPLATES_ROOT = Path(__file__).parent / "templates"


def load_prompt_yaml(domain: str, name: str) -> dict[str, str]:
    """YAML 프롬프트 파일 로드.

    Args:
        domain: 서브디렉터리명 (예: interview, chat, evaluation).
        name: 파일명(확장자 제외).

    Returns:
        {"system": "...", "human": "...", "ai": "..."} 형태.
        키가 없으면 빈 문자열로 채움.

    Raises:
        FileNotFoundError: 해당 경로에 파일이 없을 때.
    """
    path = TEMPLATES_ROOT / domain / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt YAML not found: {path}") from None

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {path}, got {type(raw)}")

    return {
        "system": raw.get("system") or "",
        "human": raw.get("human") or "",
        "ai": raw.get("ai") or "",
    }
