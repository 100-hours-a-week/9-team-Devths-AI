from fastapi import APIRouter, UploadFile, File
from app.services.model_service import predict_image
from app.schemas.prediction import PredictionResponse
from app.core.exceptions import bad_request

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file to get a prediction from the AI model.
    """
    if not file:
        raise bad_request("file_required")
    
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise bad_request("invalid_file_type", {"allowed": ["jpg", "png", "jpeg"]})
        
    file_data = await file.read()
    
    # Pass file-like object (BytesIO)
    from io import BytesIO
    result = predict_image(BytesIO(file_data))
    
    return result
