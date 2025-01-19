import os
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

class ChatRequest(BaseModel):
    dajare: str
    score: int

@router.post("/explanation")
def chat(request: ChatRequest):
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    system_content = """
あなたは優秀なダジャレ評論家です。以下を守って解説してください。

必ず特殊文字は改行のみで、改行コードも含めて出力してください。
string型のテキストで出力してください。
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": system_content
                },
                {
                    "role" : "user",
                    "content": f"「{request.dajare}」という100点中{request.score}点のダジャレについて、箇条書きで150字以内で解説してください"
                }
            ]
        )

        answer = response.choices[0].message.content

        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}