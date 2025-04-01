import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()
            positive_score = probs[1].item()
            negative_score = probs[0].item()

        label = "Positive" if positive_score > negative_score else "Negative"
        return label, positive_score, negative_score
