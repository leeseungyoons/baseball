import streamlit as st
import re
from sentiment_model import SentimentAnalyzer
from keyword_extractor import KeywordExtractor
from headline_generator import HeadlineGenerator  # ✅ 추가
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# 인스턴스 생성
sa = SentimentAnalyzer()
ke = KeywordExtractor()
hg = HeadlineGenerator()  # ✅ 헤드라인 생성기

st.title("⚾ 스포츠 기사 분석 시스템")

uploaded_file = st.file_uploader("📰 뉴스나 중계 텍스트 파일을 넣어주세요. (.txt)", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    articles = re.split(r'(?:\n\s*){5,}', text.strip())  # 빈 줄 5줄 이상 기준 분리

    sentiment_counts = Counter({"긍정": 0, "부정": 0})
    all_keywords = Counter()

    st.subheader("📄 기사 분석 결과")
    label_map = {"Positive": "긍정", "Negative": "부정"}

    positive_words = ["승리", "대승", "완승", "홈런", "안타", "우승", "역전", "세이브", "멀티히트", "3안타", "2안타", "3연승"]
    negative_words = ["패배", "병살타", "실책", "놓쳤다", "무득점", "패전", "무승부", "무산", "부진", "역전패"]

    for idx, article in enumerate(articles):
        st.markdown(f"### 📰 기사 #{idx+1}")
        st.text(article)

        try:
            label, pos_score, neg_score = sa.predict(article)
        except Exception as e:
            st.error(f"❗ 감정 분석 오류: {e}")
            continue

        translated_label = label_map.get(label, label)
        has_positive = any(word in article for word in positive_words)
        has_negative = any(word in article for word in negative_words)

        if has_negative:
            label = "Negative"
            translated_label = "부정"
            st.caption("⚠️ 부정 키워드 포함으로 감정 결과가 보정되었습니다.")
        elif label == "Negative" and has_positive:
            label = "Positive"
            translated_label = "긍정"
            st.caption("✅ 긍정 키워드 포함으로 감정 결과가 보정되었습니다.")

        st.write(f"**감정 분석 결과:** {translated_label}")
        st.write(f"긍정: {pos_score:.4f} / 부정: {neg_score:.4f}")
        st.progress(pos_score)
        st.caption("⚠️ 감정 분석은 일반 텍스트 기반이며, 스포츠 기사에서는 실제 맥락과 다르게 분류될 수 있습니다.")

        sentiment_counts[translated_label] += 1

        # 키워드 추출 및 누적
        keywords = ke.extract(article)
        all_keywords.update(dict(keywords))

        st.write("**Top Keywords:**", ", ".join(keywords.keys()) if keywords else "없음")

        # ✅ 헤드라인 생성
        try:
            headline = hg.generate(article)
            st.write(f"📰 **자동 생성 헤드라인:** {headline}")
        except Exception as e:
            st.warning(f"헤드라인 생성 실패: {e}")

        st.markdown("---")

    st.info("ℹ️ 여러 기사를 넣으려면 기사 사이에 **빈 줄 5칸 이상** (Enter 5번)을 넣어주세요!")

    # 한글 폰트 설정
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

    # 워드클라우드
    st.subheader("☁️ 키워드 워드 클라우드")

    if all_keywords:
        wc = WordCloud(
            font_path=font_path,
            background_color="white",
            width=800,
            height=400
        )
        wc.generate_from_frequencies(dict(all_keywords))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)
    else:
        st.write("❗ 키워드가 충분하지 않아 워드클라우드를 생성할 수 없습니다.")
