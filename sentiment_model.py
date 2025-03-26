# sentiment_model.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            return "긍정" if label == 1 else "부정", probs[0][label].item()
