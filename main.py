# main.py
from sentiment_model import SentimentAnalyzer
from keyword_extractor import KeywordExtractor

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    text = load_text("sample_input.txt")
    
    print("📰 입력 텍스트:")
    print(text)
    print("-" * 50)
    
    # 감정 분석
    sa = SentimentAnalyzer()
    label, prob = sa.predict(text)
    print(f"감정 분석 결과: {label} (확률: {prob:.2f})")

    # 키워드 추출
    ke = KeywordExtractor()
    keywords = ke.extract(text)
    print("주요 키워드:")
    for word, count in keywords:
        print(f"- {word} ({count}회)")
