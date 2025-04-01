import streamlit as st
from sentiment_model import SentimentAnalyzer
from keyword_extractor import KeywordExtractor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


# 모델 및 키워드 추출기 인스턴스
sa = SentimentAnalyzer()
ke = KeywordExtractor()

st.title("⚾ 스포츠 기사 분석 시스템")

uploaded_file = st.file_uploader("📰 뉴스나 중계 텍스트 파일을 넣어주세요. (.txt)", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    articles = text.strip().split("\n\n")  # 문단 기준 나누기

    sentiment_counts = Counter()
    all_keywords = Counter()

    st.subheader("📄 Article Analysis Result")
    for idx, article in enumerate(articles):
        st.markdown(f"### 📰 Article #{idx+1}")
        st.text(article)

        label, prob = sa.predict(article)
        st.write(f"**Sentiment:** {label} (Confidence: {prob:.2f})")
        sentiment_counts[label] += 1

        keywords = ke.extract(article)
        all_keywords.update(dict(keywords))
        st.write("**Top Keywords:**", ", ".join([k for k, _ in keywords]))
        st.markdown("---")

font_path = "NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

st.subheader("📊 감정 분석 요약")

labels = list(sentiment_counts.keys())  # 예: ["긍정", "부정"]
values = list(sentiment_counts.values())
colors = ["#4da6ff" if l == "긍정" else "#ff6666" for l in labels]

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["긍정", "부정"], [5, 3], color=["#4da6ff", "#ff6666"])

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{int(height)}",
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, max(values) + 1)
ax.set_ylabel("문서 수", fontsize=11, fontproperties=font_prop)
ax.set_xlabel("감정 분류", fontsize=11, fontproperties=font_prop)
ax.set_title("감정 분석 결과 분포", fontsize=14, fontweight='bold', fontproperties=font_prop)

st.pyplot(fig)

# ☁️ Keyword WordCloud
st.subheader("☁️키워드 워드 클라우드")

wc = WordCloud(
    font_path="NanumGothic.ttf",  
    background_color="white",
    width=800,
    height=400
)
wc.generate_from_frequencies(all_keywords)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.imshow(wc, interpolation="bilinear")
ax2.axis("off")
st.pyplot(fig2)
