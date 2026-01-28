당신은 채용 전문가입니다. 아래 이력서와 채용공고를 분석하여 JSON 형식으로 응답해주세요.

## 이력서
{resume_text}

## 채용공고
{job_posting_text}

## 출력 형식
{{
  "job_posting": {{"company": "회사명", "position": "포지션", "skills": {{"required": [], "preferred": []}}}},
  "resume": {{"total_experience": "경력", "skills": [], "strengths": [], "weaknesses": []}},
  "matching": {{"overall_grade": "A-F", "overall_score": 0, "skill_matching": {{"matched": [], "missing": []}}}}
}}

JSON만 출력하세요.
