# ADR-042: OCR 전략 (미완료)

> **작성 상태**: 🚧 초안 작성 중  
> **최종 업데이트**: 2026-01-19  
> **상태**: Accepted

---

## Context (배경)

PDF/이미지에서 텍스트를 추출하는 OCR 전략을 결정해야 합니다.

### 요구사항
- **한국어 정확도**: 95% 이상
- **처리 속도**: 페이지당 3초 이내
- **비용**: 월 $50 이하 (1,000건 기준)

### 후보 솔루션
1. **pytesseract** (Tesseract OCR, 무료)
2. **Gemini Vision API** (Google, 유료)

---

## Decision (결정)

### 하이브리드 전략

#### vLLM 모드 (가성비):
```
PDF → pytesseract OCR → 텍스트 추출 → VectorDB
```

#### Gemini 모드 (프리미엄):
```
PDF → Gemini Vision API → 텍스트 추출 + 레이아웃 분석 → VectorDB
```

### 버전별 전략

| 버전 | OCR 엔진 | 근거 |
|------|---------|------|
| **V1-V2** | Gemini Vision API | 빠른 프로토타이핑, 정확도 우선 |
| **V3** | pytesseract (vLLM 모드) | 비용 절감 |
| **V3** | Gemini Vision (Gemini 모드) | 프리미엄 옵션 |

---

## Consequences (결과)

### 긍정적 영향
- ✅ 사용자 선택권 제공 (가성비 vs 프리미엄)
- ✅ 비용 절감 (pytesseract 무료)
- ✅ 정확도 보장 (Gemini Vision 옵션)

### 부정적 영향
- ❌ pytesseract 정확도 제한적 (~90%)
- ❌ 두 가지 OCR 로직 유지보수 필요

### 완화 전략
- pytesseract 전처리 최적화 (이미지 해상도, 노이즈 제거)
- Fallback 구현 (pytesseract 실패 시 Gemini로 전환)

---

## Performance Comparison (성능 비교)

| OCR 엔진 | 한국어 CER | 처리 속도 | 월간 비용 (1,600 페이지) |
|---------|-----------|---------|------------------------|
| **pytesseract** | 10% | 2.5초/페이지 | $0 |
| **Gemini Vision** | 3% | 1.5초/페이지 | $4.00 |

**결론**: Gemini Vision이 정확도와 속도 모두 우수하지만, pytesseract도 충분한 성능

---

## Alternatives Considered (고려한 대안)

### 1. Naver Clova OCR
- **장점**: 한국어 특화, 높은 정확도
- **단점**: API 비용, 국내 서비스 제한
- **결론**: 비용 대비 Gemini Vision이 유리

### 2. AWS Textract
- **장점**: 표 구조 인식 우수
- **단점**: 한국어 지원 제한적
- **결론**: 한국어 성능 부족

---

## Implementation (구현)

### pytesseract 사용 예시
```python
import pytesseract
from PIL import Image

# 한국어 + 영어 OCR
text = pytesseract.image_to_string(
    Image.open('resume.jpg'),
    lang='kor+eng'
)
```

### Gemini Vision 사용 예시
```python
import google.generativeai as genai

model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content([
    "Extract all text from this image",
    image_data
])
text = response.text
```

---

## Related ADRs
- ADR-040: LLM 선정
- ADR-044: Masking 전략

---

## References
- [Tesseract OCR Documentation](https://github.com/tesseract-ocr/tesseract)
- [Gemini Vision API](https://ai.google.dev/gemini-api/docs/vision)
- [Model Select/02_OCR_모델_선정.md](../Model%20Select/02_OCR_모델_선정(미완료).md)
