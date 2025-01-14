# ニューラルネットワークモデルのクラス定義
class DajarePredictor(nn.Module):
    def __init__(self, bert_model):
        super(DajarePredictor, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS]トークンの埋め込み
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x