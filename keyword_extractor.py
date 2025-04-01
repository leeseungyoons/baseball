from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class KeywordExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp_ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def extract(self, text):
        ner_results = self.nlp_ner(text)
        entities = {"PER": [], "ORG": [], "RECORD": []}
        for entity in ner_results:
            if "PER" in entity["entity"]:
                entities["PER"].append(entity["word"])
            elif "ORG" in entity["entity"]:
                entities["ORG"].append(entity["word"])
            elif "안타" in entity["word"] or "홈런" in entity["word"]:
                entities["RECORD"].append(entity["word"])
        return entities
