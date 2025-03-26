# keyword_extractor.py
from konlpy.tag import Okt
from collections import Counter

class KeywordExtractor:
    def __init__(self):
        self.okt = Okt()

    def extract(self, text, top_k=5):
        words = self.okt.nouns(text)
        words = [w for w in words if len(w) > 1]  # 1글자 제거
        count = Counter(words)
        return count.most_common(top_k)
