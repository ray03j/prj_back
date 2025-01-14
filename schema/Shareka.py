import MeCab
from collections import Counter

# ダジャレかどうか判別するAI
# 以下の記事を参照
# https://qiita.com/fujit33/items/dbfbd7a2aa3858067b6c#shareka%E3%83%80%E3%82%B8%E3%83%A3%E3%83%AC%E5%88%A4%E5%88%A5ai
class Shareka:
    def __init__(self, sentence, n=2):
        """置き換える文字リストが格納されたクラス変数"""
        self.replace_words = [["。", ""], ["、", ""], [",", ""], [".", ""], ["!", ""],
                              ["！", ""], ["・", ""], ["「", ""], ["」", ""], ["｣", ""],
                              ["『", ""], ["』", ""], [" ", ""], ["　", ""],
                              ["ッ", ""], ["ャ", "ヤ"], ["ュ", "ユ"], ["ョ", "ヨ"],
                              ["ァ", "ア"], ["ィ", "イ"], ["ゥ", "ウ"], ["ェ", "エ"], ["ォ", "オ"], ["ー", ""], ["ヲ", "オ"]]
        self.kaburi = n
        self.sentence = sentence

        mecab = MeCab.Tagger("-Oyomi")
        self.kana = mecab.parse(sentence).strip()  # 改行を削除
        self.preprocessed = self.preprocessing(self.kana)
        self.devided = self.devide(self.preprocessed)

    def preprocessing(self, sentence):
        """文字列を置き換える前処理"""
        for replace_word in self.replace_words:
            sentence = sentence.replace(replace_word[0], replace_word[1])
        return sentence

    def devide(self, sentence):
        """文字列を指定された長さで分割"""
        elements = []
        repeat_num = len(sentence) - (self.kaburi - 1)
        for i in range(repeat_num):
            elements.append(sentence[i:i+self.kaburi])
        return elements

    def has_duplicates(self):
        """重複する部分文字列があるかどうかを判定"""
        if not self.devided:
            return False
        counter = Counter(self.devided)
        for value in counter.values():
            if value > 1:
                return True
        return False

    def dajarewake(self):
        """駄洒落判定"""
        return self.has_duplicates()
