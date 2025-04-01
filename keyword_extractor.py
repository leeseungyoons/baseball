from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class KeywordExtractor:
    def __init__(self):
        # NER 모델 불러오기
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base-ner")
        self.model = AutoModelForTokenClassification.from_pretrained("klue/bert-base-ner")

        self.nlp_ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def extract(self, text):
        # NER을 통한 개체명 추출
        ner_results = self.nlp_ner(text)
        
        entities = {"PER": [], "ORG": [], "RECORD": []}

        for entity in ner_results:
            if entity["entity"] == "B-PER":
                entities["PER"].append(entity["word"])  # 선수명
            elif entity["entity"] == "B-ORG":
                entities["ORG"].append(entity["word"])  # 팀명
            elif "기록" in entity["word"]:  # 예시: 기록 포함 키워드
                entities["RECORD"].append(entity["word"])

        return entities
