from transformers import AutoTokenizer, AutoModel
import torch

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Load model (only encoder, no classification head)
model = AutoModel.from_pretrained("bert-base-uncased")

# 3. Text input
text = "I love AI"

# 4. Tokenize
inputs = tokenizer(text, return_tensors="pt")

# 5. Pass through model
outputs = model(**inputs)

# 6. Get embeddings
embeddings = outputs.last_hidden_state

print("Embedding shape:", embeddings.shape)

# the embedding has shape which is [batch size , number of tokens , embedding size(768)] 
