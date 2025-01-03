import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path



def explanation_dajare(dajare, dajare_kana, model="gpt-4o-mini"):
    # 親ディレクトリのパスを取得
    parent_dir = Path(__file__).resolve().parent.parent

    # 親ディレクトリにある .env ファイルのパスを指定
    env_path = parent_dir / '.env'

    # .env ファイルを読み込む
    load_dotenv(dotenv_path=env_path)

    # 環境変数を取得
    api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI()

    system_content = """
あなたは優秀なAIアシスタントです。以下の観点で
1. 変形表現に促音、拗音が入っている：例えば、「内股を打っちまったー」が「うちまた」が「うっちまったー」と促音が含まれている形に変形している
2. 母音が同じである（全く同じ読みも含む）：例えば、「公園に誰もこうへん」は「おうえん」という母音が同じ
3. 母音が1音変化している（1音挿入/脱落も含む）：例えば、「横須賀(よこすか)は要(よう)港(こう)すか」は「よこすか」から「ようこうすか」に変化している
4. 方言である：例えば、「このつくね、いつ食うねん？」は関西弁
5. 種表現の別の呼び方を使用している：例えば、「お金を落としてすマネー」はお金をマネーと言い換えている
6. 場面をイメージしやすい・わかりやすい例えば、「坊ちゃんがぼっちゃんと川に落ちる」は男の子が川に落ちるというイメージがわかりやすい
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role" : "user",
                "content": f"「{dajare}」というダジャレについて、箇条書きで150字以内で解説してください"
            }
        ]
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    dajare = "おにぎりで、鬼斬り ！！"
    dajare_kana = "おにぎりでおにぎり"
    print(explanation_dajare(dajare, dajare_kana))

