from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_texts(texts):
    return tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=32, 
        return_tensors="pt"
    )
