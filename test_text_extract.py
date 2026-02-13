"""
Test script for /ai/text/extract endpoint
"""
import json

import requests

# Test data
test_request = {
    "model": "gemini",
    "user_id": 12,
    "resume": {
        "text": "이름: 홍길동\n경력: 3년\n기술스택: Python, FastAPI, React"
    },
    "job_posting": {
        "text": "카카오 백엔드 개발자 채용\n자격요건: Python, FastAPI 경험자"
    }
}

# API endpoint
url = "http://localhost:8000/ai/text/extract"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key-here"
}

# [Test Safety] 테스트 수집 시(pytest collection) 이 코드가 실행되는 것을 방지하기 위해 main 블록으로 감쌉니다.
if __name__ == "__main__":
    print("📤 Sending request to /ai/text/extract...")
    print(f"Request body:\n{json.dumps(test_request, indent=2, ensure_ascii=False)}\n")

    try:
        response = requests.post(url, json=test_request, headers=headers)
        print(f"✅ Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 202:
            task_id = response.json().get("task_id")
            print(f"\n📋 Task ID: {task_id}")
            print(f"💡 Poll status at: GET {url.replace('/text/extract', f'/task/{task_id}')}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError:
        print("❌ Response is not JSON:")
        print(response.text)
