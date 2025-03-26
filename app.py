# app.py
import streamlit as st
from sentiment_model import SentimentAnalyzer
from keyword_extractor import KeywordExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# 인스턴스 생성
sa = SentimentAnalyzer()
ke = KeywordExtractor()

st.title("⚾ 야구 뉴스 감정 분석 & 키워드 추출기")

uploaded_file = st.file_uploader("📰 뉴스 텍스트 파일 업로드 (.txt)", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    articles = text.strip().split("\n\n")  # 문단 분리

    sentiment_counts = Counter()
    all_keywords = Counter()

    st.subheader("📄 뉴스 문서 분석 결과")
    for idx, article in enumerate(articles):
        st.markdown(f"### 📰 뉴스 #{idx+1}")
        st.text(article)

        label, prob = sa.predict(article)
        st.write(f"**감정 분석 결과:** {label} (신뢰도: {prob:.2f})")
        sentiment_counts[label] += 1

        keywords = ke.extract(article)
        all_keywords.update(dict(keywords))
        st.write("**주요 키워드:**", ", ".join([k for k, _ in keywords]))
        st.markdown("---")

    # ✅ 감정 분석 요약 그래프
    st.subheader("📊 감정 분석 요약")

    # 💡 폰트 설정: 프로젝트 폴더에 있는 NanumGothic.ttf 사용
    font_path = "NanumGothic.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    colors = ["#4da6ff" if l == "긍정" else "#ff6666" for l in labels]

    bars = ax.bar(labels, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{int(height)}",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, max(values) + 1)
    ax.set_ylabel("문서 수", fontsize=11)
    ax.set_xlabel("감정 분류", fontsize=11)
    ax.set_title("감정 분석 결과 분포", fontsize=14, weight='bold')

    st.pyplot(fig)

    # ✅ 워드클라우드
    st.subheader("☁️ 전체 키워드 워드클라우드")
    wc = WordCloud(font_path=font_path, background_color="white", width=800, height=400)
    wc.generate_from_frequencies(all_keywords)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)
