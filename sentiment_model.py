# sentiment_model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self):
        # KLUE-BERT 기반 이진 감정 분류 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained("model")
        self.model = AutoModelForSequenceClassification.from_pretrained("model")

        self.model.eval()

    def predict(self, text):
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        # 확률 점수 추출
        positive_score = probs[1].item()  # index 1 = positive
        negative_score = probs[0].item()  # index 0 = negative

        # 확률 기반으로 감정 라벨 직접 설정
        if positive_score > negative_score:
            label = "Positive"
        else:
            label = "Negative"

        return label, positive_score, negative_score
