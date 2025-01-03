import os
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

class ChatRequest(BaseModel):
    dajare: str

@router.post("/settai")
def chat(request: ChatRequest):
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    system_content = """
    あなたは優秀な後輩です。相手が喜ぶ言葉遣いを知っていて、適切な言葉を選ぶことができます。
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
                    "content": f"「{request.dajare}」というダジャレについて、セリフ調で大げさに150字以内で褒めてください"
                }
            ]
        )

        answer = response.choices[0].message.content

        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}