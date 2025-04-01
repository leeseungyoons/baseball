# sentiment_model.py

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        model_name = "klue/bert-base"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()

        # 클래스 순서가 ["Negative", "Positive"]라고 가정 (필요 시 확인)
        self.label_map = {0: "부정", 1: "긍정"}

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()

        label_idx = torch.argmax(probs).item()
        label = self.label_map[label_idx]
        return {
            "label": label,
            "positive_prob": round(probs[1].item(), 4),
            "negative_prob": round(probs[0].item(), 4)
        }
