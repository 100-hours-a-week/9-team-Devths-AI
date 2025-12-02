from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str
    confidence_score: float
