# sentiment_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "klue/bert-base", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs).item()
            return ("Positive" if label == 1 else "Negative", probs[0][label].item())
