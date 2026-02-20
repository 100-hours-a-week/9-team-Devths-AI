"""
EasyOCR Server (v2) - OCR 보조 서버.
엔드포인트: POST /ocr/extract
"""
import logging
from typing import Any

import easyocr
from fastapi import FastAPI, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

app = FastAPI(title="EasyOCR Server", version="1.0.0")

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(["ko", "en"], gpu=True)
            logger.info("EasyOCR Reader initialized (GPU)")
        except Exception as e:
            logger.warning(f"EasyOCR GPU failed, using CPU: {e}")
            _reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _reader


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "easyocr-server"}


@app.post("/ocr/extract")
async def ocr_extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """이미지에서 텍스트 추출 (EasyOCR)."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        reader = get_reader()
        result = reader.readtext(contents)
        text = " ".join([r[1] for r in result])
        return {"text": text, "success": True}
    except Exception as e:
        logger.exception("OCR failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
