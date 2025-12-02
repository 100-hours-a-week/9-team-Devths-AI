from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3:4b"
