from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import generate_chat_response

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_chat_response(request.message, request.model),
        media_type="application/x-ndjson"
    )
