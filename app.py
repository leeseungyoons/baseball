import streamlit as st
import re
from sentiment_model import SentimentAnalyzer
from keyword_extractor import KeywordExtractor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# 모델 인스턴스
sa = SentimentAnalyzer()
ke = KeywordExtractor()

st.title("⚾ 스포츠 기사 분석 시스템")

uploaded_file = st.file_uploader("📰 뉴스나 중계 텍스트 파일을 넣어주세요. (.txt)", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    articles = re.split(r'\n\s*\n', text.strip())  # 줄 2개 이상을 기사 구분 기준으로

    # 기본값 포함된 감정 카운터
    sentiment_counts = Counter({"긍정": 0, "부정": 0})
    all_keywords = Counter()

    st.subheader("📄 기사 분석 결과")
    label_map = {"Positive": "긍정", "Negative": "부정"}
    positive_words = ["승리", "대승", "완승", "홈런", "안타", "우승", "역전", "세이브", "멀티히트", "3안타", "2안타", "3연승"]

    for idx, article in enumerate(articles):
        st.markdown(f"### 📰 기사 #{idx+1}")
        st.text(article)

        # 감정 분석
        label, prob = sa.predict(article)
        translated_label = label_map.get(label, label)

        # 🔧 보정: 부정인데 긍정 키워드가 포함돼 있으면 긍정으로 변경
        if label == "Negative" and any(word in article for word in positive_words):
            label = "Positive"
            translated_label = "긍정"
            st.caption("✅ 스포츠 긍정 키워드가 포함되어 있어 감정 결과가 보정되었습니다.")

        # 출력
        st.write(f"**감정 분석 결과:** {translated_label} (신뢰도: {prob:.2f})")

        # 🔹 감정 설명
        st.caption("⚠️ 감정 분석은 일반 텍스트 데이터 기반이며, 스포츠 기사에서는 실제 맥락과 다르게 분류될 수 있습니다.")

        # 카운트 반영
        sentiment_counts[translated_label] += 1

        # 키워드 추출
        keywords = ke.extract(article)
        all_keywords.update(dict(keywords))
        st.write("**Top Keywords:**", ", ".join([k for k, _ in keywords]))
        st.markdown("---")

    # ✅ 폰트 설정
    font_path = "NanumGothic.ttf"
    font_prop = fm.FontProperties(fname=font_path)

    st.subheader("📊 감정 분석 요약")

    labels = ["긍정", "부정"]
    values = [sentiment_counts["긍정"], sentiment_counts["부정"]]
    colors = ["#4da6ff", "#ff6666"]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{int(height)}",
                ha='center', va='bottom', fontsize=12, fontweight='bold', fontproperties=font_prop)

    ax.set_xticklabels(labels, fontproperties=font_prop)
    ax.set_ylim(0, max(values) + 1)
    ax.set_ylabel("감정별 기사 개수", fontsize=11, fontproperties=font_prop)
    ax.set_xlabel("감정 분류", fontsize=11, fontproperties=font_prop)
    ax.set_title("감정 분석 결과 분포", fontsize=14, fontweight='bold', fontproperties=font_prop)

    st.pyplot(fig)

    # ☁️ 키워드 워드클라우드
    st.subheader("☁️ 키워드 워드 클라우드")

    if all_keywords:
        wc = WordCloud(
            font_path=font_path,
            background_color="white",
            width=800,
            height=400
        )
        wc.generate_from_frequencies(all_keywords)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)
    else:
        st.write("❗ 키워드가 충분하지 않아 워드클라우드를 생성할 수 없습니다.")
