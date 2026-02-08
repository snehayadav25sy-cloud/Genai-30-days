from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love AI"
tokens = tokenizer(text)

print(tokens)

#after converting it into tokens , it has token id , which represent the position of each token
