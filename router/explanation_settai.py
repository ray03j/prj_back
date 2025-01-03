import os
from fastapi import APIRouter
from pydantic import BaseModel
# from openai import OpenAI
import openai

router = APIRouter()

class ChatRequest(BaseModel):
    model: str
    messages: list

@router.post("/settai")
def chat(request: ChatRequest):
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    system_content = """
    """

    response = client.chat.completions.create(
        model=request.model,
        messages=[
            {
                "role": "system", 
                "content": system_content
            },
            *request.messages
        ]
    )
    return response