# keyword_extractor.py
import re
from collections import Counter

class KeywordExtractor:
    def extract(self, text):
        # 한글 2글자 이상인 단어만 추출
        words = re.findall(r"[가-힣]{2,}", text)

        # 너무 흔한 불용어 제거
        common_words = ["기자", "입니다", "했다", "한다", "있는", "있다", "이번", "그리고", "대해"]
        words = [w for w in words if w not in common_words]

        # 빈도수 계산
        freq = Counter(words)
        return freq.most_common(10)  # 상위 10개만 리턴
