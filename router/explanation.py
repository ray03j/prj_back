import os
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

class ChatRequest(BaseModel):
    dajare: str

@router.post("/explanation")
def chat(request: ChatRequest):
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    system_content = """
あなたは優秀なダジャレ評論家です。以下の観点で解説してください。
1. 変形表現に促音、拗音が入っている：例えば、「内股を打っちまったー」が「うちまた」が「うっちまったー」と促音が含まれている形に変形している
2. 母音が同じである（全く同じ読みも含む）：例えば、「公園に誰もこうへん」は「おうえん」という母音が同じ
3. 母音が1音変化している（1音挿入/脱落も含む）：例えば、「横須賀(よこすか)は要(よう)港(こう)すか」は「よこすか」から「ようこうすか」に変化している
4. 方言である：例えば、「このつくね、いつ食うねん？」は関西弁
5. 種表現の別の呼び方を使用している：例えば、「お金を落としてすマネー」はお金をマネーと言い換えている
6. 場面をイメージしやすい・わかりやすい例えば、「坊ちゃんがぼっちゃんと川に落ちる」は男の子が川に落ちるというイメージがわかりやすい
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
                    "content": f"「{request.dajare}」というダジャレについて、箇条書きで150字以内で解説してください"
                }
            ]
        )

        answer = response.choices[0].message.content

        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}