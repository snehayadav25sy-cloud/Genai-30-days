documents = [
    "Artificial intelligence is transforming industries.",
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "FAISS enables fast similarity search.",
    "Vector databases store embeddings.",
    "Natural language processing works with text.",
    "Retrieval augmented generation improves LLM accuracy.",
]

#Convert to Embeddings

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(documents)

embeddings=np.array(embeddings).astype("float32")


#Build FAISS Index
dimension=embeddings.shape[1]

index=faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("total vectors:",index.ntotal)

#Query Search
query="how does ai use neural networks ?"
query_emd=model.encode([query])
query_emd = np.array(query_emd).astype("float32")

k=3
distances,indices=index.search(query_emd,k)

print("top results:")
for idx in indices[0]:
    print(documents[idx])