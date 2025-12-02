from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.services.sentiment_service import SentimentAnalysisService, get_sentiment_service
from app.core.exceptions import bad_request, unprocessable

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="분석할 텍스트")
    explain: bool = Field(default=False, description="토큰 영향도 포함 여부")


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    top_tokens: list


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    payload: SentimentRequest,
    service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """
    텍스트 감성 분석 API
    - text: 분석할 영어 텍스트
    - explain: True이면 토큰별 영향도 포함
    """
    try:
        if not payload.text or not payload.text.strip():
            raise bad_request("text_required")
        
        result = service.predict(payload.text)
        
        # explain이 False면 top_tokens 제거
        if not payload.explain:
            result["top_tokens"] = []
        
        return result
        
    except ValueError as e:
        raise bad_request(str(e))
    except Exception as e:
        raise unprocessable("sentiment_analysis_failed", {"details": str(e)})


