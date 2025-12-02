import os
import numpy as np
from PIL import Image, ImageOps
# from keras.models import load_model  # Moved to lazy load
from app.core.exceptions import internal_server_error, unprocessable

# Global variables to hold model and labels
MODEL = None
CLASS_NAMES = []

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(ASSETS_DIR, "labels.txt")

def load_ai_model():
    """Load model and labels into global variables."""
    global MODEL, CLASS_NAMES
    try:
        # TensorFlow를 먼저 import하여 백엔드 초기화
        import tensorflow as tf
        print(f"TensorFlow 버전: {tf.__version__}")
        
        from keras.models import load_model
        print(f"Loading model from {MODEL_PATH}...")
        MODEL = load_model(MODEL_PATH, compile=False)
        
        print(f"Loading labels from {LABELS_PATH}...")
        with open(LABELS_PATH, "r") as f:
            CLASS_NAMES = [line.strip() for line in f.readlines()]
            
        print("✅ Model and labels loaded successfully.")
    except ImportError as e:
        print(f"❌ Import 에러: {e}")
        print("TensorFlow/Keras가 설치되어 있는지 확인하세요: pip install tensorflow-macos tensorflow-metal keras")
        MODEL = None
        CLASS_NAMES = []
    except Exception as e:
        print(f"❌ 모델 로딩 에러: {e}")
        MODEL = None
        CLASS_NAMES = []

def predict_image(image_file) -> dict:
    """
    Predict the class of the image.
    Args:
        image_file: file-like object (bytes)
    Returns:
        dict: {"class_name": str, "confidence_score": float}
    """
    global MODEL, CLASS_NAMES
    
    if MODEL is None or not CLASS_NAMES:
        # Try loading again if not loaded
        load_ai_model()
        if MODEL is None:
            raise internal_server_error("model_not_loaded")

    try:
        # Open image
        image = Image.open(image_file).convert("RGB")
        
        # Resize and preprocess
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        
        # Normalize
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Prepare data array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Predict
        prediction = MODEL.predict(data)
        index = np.argmax(prediction)
        class_name = CLASS_NAMES[index]
        confidence_score = float(prediction[0][index])
        
        # Clean class name (remove index if present, e.g., "0 Cat" -> "Cat")
        # The reference code did class_name[2:], assuming "0 " prefix.
        # We'll be safer.
        if " " in class_name:
            class_name = class_name.split(" ", 1)[1]
            
        return {
            "class_name": class_name,
            "confidence_score": confidence_score
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise unprocessable("prediction_failed", {"details": str(e)})
