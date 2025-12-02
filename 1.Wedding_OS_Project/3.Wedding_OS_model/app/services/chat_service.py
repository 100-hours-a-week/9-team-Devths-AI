from ollama import chat
import json
from typing import AsyncGenerator

async def generate_chat_response(message: str, model: str = "gemma3:4b") -> AsyncGenerator[str, None]:
    stream = chat(
        model=model,
        messages=[{'role': 'user', 'content': message}],
        stream=True,
    )
    
    in_thinking = False
    thinking = ''
    content = ''

    for chunk in stream:
        data = {}
        
        # Handle thinking
        if chunk.message.thinking:
            if not in_thinking:
                in_thinking = True
                data['type'] = 'thinking_start'
                yield json.dumps(data) + "\n"
            
            data = {
                'type': 'thinking',
                'content': chunk.message.thinking
            }
            thinking += chunk.message.thinking
            yield json.dumps(data) + "\n"
        
        # Handle content
        elif chunk.message.content:
            if in_thinking:
                in_thinking = False
                data = {'type': 'thinking_end'}
                yield json.dumps(data) + "\n"
            
            data = {
                'type': 'content',
                'content': chunk.message.content
            }
            content += chunk.message.content
            yield json.dumps(data) + "\n"
