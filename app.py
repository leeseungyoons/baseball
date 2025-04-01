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
    articles = re.split(r'(?:\n\s*){5,}', text.strip())  # 5줄 이상 빈줄 기준 분리

    sentiment_counts = Counter({"긍정": 0, "부정": 0})
    all_keywords = Counter()

    st.subheader("📄 기사 분석 결과")
    label_map = {"Positive": "긍정", "Negative": "부정"}

    # ✅ 스포츠 긍정/부정 키워드
    positive_words = [
        "승리", "완승", "대승", "압승", "이겼다", "이기며", "역전승", "끝내기",
        "우승", "연승", "홈런", "멀티히트", "3안타", "4안타", "쾌조", "호투",
        "호성적", "활약", "맹활약", "쾌투", "무실점", "세이브", "QS", "피칭",
        "3연승", "4연승", "경기 승리", "시즌 첫 승", "첫 승리", "타점", "수훈",
        "기록 경신", "신기록", "선발 출전", "타격감", "좋은 경기력", "정상 복귀",
        "복귀전 승리", "승부처 장악", "기세", "에이스", "주전 복귀", "첫 홈런"
    ]

    negative_words = [
        "패배", "역전패", "완패", "참패", "무득점", "실책", "병살타", "부상", "이탈",
        "결장", "낙마", "부진", "부진한", "불안", "실망", "탈락", "무기력",
        "타격 침묵", "타선 침묵", "성적 부진", "ERA 상승", "타율 하락",
        "조기 강판", "볼넷", "난조", "난타", "실점", "기회 무산", "주루사",
        "연패", "3연패", "4연패", "5연패", "1군 제외", "등록 말소", "방망이도 잡지 못해",
        "엔트리 제외", "못하는", "허용", "불펜 난조", "선발 무너짐", "무승부",
        "아쉬운", "초반에 밀렸다", "판정 논란", "징계", "출전 정지"
    ]

    for idx, article in enumerate(articles):
        st.markdown(f"### 📰 기사 #{idx+1}")
        st.text(article)

        try:
            label, pos_score, neg_score = sa.predict(article)
        except Exception as e:
            st.error(f"감정 분석 오류 발생: {e}")
            continue

        translated_label = label_map.get(label, label)

        # 결과 출력
        st.write(f"**감정 분석 결과:** {translated_label}")
        st.write(f"긍정: {pos_score:.4f} / 부정: {neg_score:.4f}")
        st.progress(pos_score)
        st.caption("⚠️ 감정 분석은 일반 텍스트 기반이며, 스포츠 기사에서는 실제 맥락과 다르게 분류될 수 있습니다.")

        # 스포츠 키워드 기반 보정 메시지 (label 자체는 변경하지 않음)
        has_positive = any(word in article for word in positive_words)
        has_negative = any(word in article for word in negative_words)

        if has_negative:
            st.caption("⚠️ 부정적인 스포츠 키워드가 포함되어 있어 감정 결과가 보정될 수 있습니다.")
        elif label == "Negative" and has_positive:
            st.caption("✅ 긍정적인 스포츠 키워드가 포함되어 있어 감정 결과가 보정될 수 있습니다.")

        sentiment_counts[translated_label] += 1

        # 개체명 인식
        try:
            entities = ke.extract(article)
            st.write("📎 **개체명 추출 결과:**")
            if entities["PER"]:
                st.markdown(f"- 선수: {', '.join(set(entities['PER']))}")
            if entities["ORG"]:
                st.markdown(f"- 팀: {', '.join(set(entities['ORG']))}")
            if entities["RECORD"]:
                st.markdown(f"- 기록: {', '.join(set(entities['RECORD']))}")

            # 키워드 누적 저장 (선수명 + 팀명 기반)
            keywords = dict(Counter(entities["PER"] + entities["ORG"]))
            all_keywords.update(keywords)

        except Exception as e:
            st.error(f"개체명 추출 오류 발생: {e}")

        st.markdown("---")

    st.info("ℹ️ 여러 기사를 넣으려면 기사 사이에 **빈 줄 5칸 이상** (Enter 5번)을 넣어주세요!")

    # ✅ 한글 폰트 설정
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

    # ☁️ 워드클라우드
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
