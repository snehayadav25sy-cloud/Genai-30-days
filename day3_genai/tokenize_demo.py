from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love AI"
tokens = tokenizer(text)

print(tokens)