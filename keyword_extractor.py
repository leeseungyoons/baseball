# keyword_extractor.py (Streamlit Cloud용 경량 버전)
from collections import Counter
import re

class KeywordExtractor:
    def __init__(self):
        self.stopwords = set(["경기", "선수", "오늘", "어제", "이번", "상대", "팬", "이닝", "결과", "기록"])

    def extract(self, text, top_k=5):
        # 2글자 이상 한글 단어만 추출
        words = re.findall(r'\b[가-힣]{2,}\b', text)
        words = [w for w in words if w not in self.stopwords]
        count = Counter(words)
        return count.most_common(top_k)
