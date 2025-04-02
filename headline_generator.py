# headline_generator.py
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

class HeadlineGenerator:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-news")
        self.model = BartForConditionalGeneration.from_pretrained("digit82/kobart-news")

    def generate(self, article_text):
        input_ids = self.tokenizer.encode(article_text, return_tensors="pt", max_length=512, truncation=True)
        output = self.model.generate(input_ids, max_length=30, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
