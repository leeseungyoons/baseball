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

st.title("⚾ 스포츠 기사 감정 분석기")

uploaded_file = st.file_uploader("📰 뉴스나 중계 텍스트 파일을 넣어주세요 (.txt)", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    articles = re.split(r'(?:\n\s*){5,}', text.strip())  # 5줄 이상 빈줄 기준 분리

    sentiment_counts = Counter({"긍정": 0, "부정": 0})
    all_keywords = Counter()
    label_map = {"Positive": "긍정", "Negative": "부정"}

    # ✅ 확장된 키워드
    positive_words = [
        "승리", "대승", "역전승", "홈런", "활약", "맹타", "결승타", "극적인", "끝내기", "무실점",
        "이기며", "완봉", "4연승", "5연승", "호투", "기세", "눈부신", "연승", "역전극", "쾌조",
        "드라마틱", "MVP", "선발승", "기록 경신", "놀라운", "완벽한", "압도적", "4타수 4안타"
    ]
    negative_words = [
        "패배", "병살타", "실책", "놓쳤다", "무득점", "패전", "무승부", "무산", "부진", "역전패",
        "부상", "이탈", "불안", "혹사", "부진한 흐름", "기회 놓쳤다", "탈락", "결장", "퇴장"
    ]

    st.subheader("📄 기사 분석 결과")

    for idx, article in enumerate(articles):
        st.markdown(f"### 📰 기사 #{idx+1}")
        st.text(article)

        # 감정 분석 실행
        try:
            orig_label, pos_score, neg_score = sa.predict(article)
        except Exception as e:
            st.error(f"감정 분석 실패: {e}")
            continue

        # 감정 보정 로직
        has_positive = any(word in article for word in positive_words)
        has_negative = any(word in article for word in negative_words)
        label = orig_label  # 기본 모델 결과
        translated_label = label_map.get(label, label)

        if has_negative and not has_positive:
            label = "Negative"
            translated_label = "부정"
            st.caption("⚠️ 부정 키워드가 포함되어 있어 감정 결과가 보정되었습니다.")
        elif has_positive and not has_negative:
            label = "Positive"
            translated_label = "긍정"
            st.caption("✅ 긍정 키워드만 있어 감정 결과가 보정되었습니다.")
        elif has_positive and has_negative:
            st.caption("ℹ️ 긍정/부정 키워드가 모두 있어 모델 원래 결과 유지됨.")

        # 감정 출력
        st.write(f"**감정 분석 결과:** {translated_label}")
        st.write(f"긍정: {pos_score:.4f} / 부정: {neg_score:.4f}")
        st.progress(pos_score)
        st.caption("⚠️ 감정 분석은 일반 텍스트 기반이며, 스포츠 기사에서는 실제 맥락과 다를 수 있습니다.")

        # 카운트 저장
        sentiment_counts[translated_label] += 1

        # 키워드 추출 & 누적
        keywords = ke.extract(article)

        if keywords:
            all_keywords.update(keywords)
            st.write("**Top Keywords:**", ", ".join(keywords.keys()))
        else:
            st.write("❗ 키워드 없음")


        st.markdown("---")

    st.info("ℹ️ 여러 기사를 넣으려면 기사 사이에 **빈 줄 5칸 이상** (Enter 5번)을 넣어주세요!")

    # ✅ 한글 폰트 설정
    font_path = "NanumGothic.ttf"
    font_prop = fm.FontProperties(fname=font_path)

    # 📊 감정 요약 그래프
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

    # ☁️ 워드클라우드
    st.subheader("☁️ 키워드 워드 클라우드")

    
    
    if all_keywords:
        wc = WordCloud(
            font_path=font_path,
            background_color="white",
            width=800,
            height=400
        )

        freq_dict = dict((k, int(v)) for k, v in all_keywords.items() if isinstance(v, (int, float)) and v > 0)

        if freq_dict:
        
            wc.generate_from_frequencies(dict(all_keywords))
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")
            st.pyplot(fig2)
        else:
            st.warning("❗ 키워드가 부족해 워드클라우드를 생성할 수 없습니다.")
    else:
        st.warning("❗ 키워드가 부족해 워드클라우드를 생성할 수 없습니다.")
