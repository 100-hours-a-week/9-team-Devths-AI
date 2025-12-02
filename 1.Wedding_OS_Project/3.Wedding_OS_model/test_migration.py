import requests
import json

url = "http://localhost:8001/api/chat"
data = {
    "message": "What is 12 * 12?",
    "model": "gemma3:4b"
}

print(f"Sending request to {url}...")
try:
    with requests.post(url, json=data, stream=True) as r:
        print(f"Status Code: {r.status_code}")
        for line in r.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode('utf-8'))
                    if json_data['type'] == 'thinking':
                        print(json_data['content'], end='', flush=True)
                    elif json_data['type'] == 'content':
                        print(json_data['content'], end='', flush=True)
                    elif json_data['type'] == 'thinking_start':
                        print("\n[Thinking Start]\n")
                    elif json_data['type'] == 'thinking_end':
                        print("\n[Thinking End]\n")
                except json.JSONDecodeError:
                    print(f"\nError decoding line: {line}")
    print("\nDone.")
except Exception as e:
    print(f"Request failed: {e}")
