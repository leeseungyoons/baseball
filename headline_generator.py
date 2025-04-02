from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HeadlineGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("lcw99/t5-korean-news-headline-generation")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("lcw99/t5-korean-news-headline-generation")

    def generate(self, text, max_length=64):
        input_ids = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        output = self.model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
        headline = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return headline
