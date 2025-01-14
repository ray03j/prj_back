from fastapi import APIRouter, HTTPException
import torch
from transformers import BertJapaneseTokenizer, BertModel
import MeCab
from schema.DajarePredictor import DajarePredictor
from schema.Shareka import Shareka
import os


router = APIRouter()

# 必要な変数とパスを設定
version = "v1.14"
load_dir = f"../models/{version}"
pretrained_model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# MeCabの設定
mecab = MeCab.Tagger("-Owakati")

# モデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
bert_model = BertModel.from_pretrained(pretrained_model_name)
model = DajarePredictor(bert_model)
model_path = os.path.join(load_dir, "Dajudge_epoch_5.pth")

# モデルのパラメータをロード
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@router.get("/")
async def predict(dajare: str):
    """
    入力された文字列のダジャレスコアを計算し、クラスを返す。
    
    Parameters:
        query (str): 入力テキスト

    Returns:
        dict: スコアと予測クラス
    """
    if not dajare:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    try:
        shareka_instance = Shareka(dajare, n=3)
        is_dajare = int(shareka_instance.dajarewake())

        if is_dajare:
            # 形態素解析
            tokenized_text = mecab.parse(dajare).strip()
            # トークナイズ
            encoding = tokenizer(
                tokenized_text,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # 予測
            with torch.no_grad():
                prediction = model(input_ids, attention_mask).squeeze().item()
                prediction = torch.sigmoid(torch.tensor(prediction)).item()  # Sigmoid関数適用
                predicted_class = 1 if prediction >= 0.5 else 0  # クラス分類
        else:
            prediction = 0.0
            predicted_class = 0

        return {
            "query": dajare,
            "score": prediction,
            "predicted_class": "Funny" if predicted_class == 1 else "Not Funny"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

