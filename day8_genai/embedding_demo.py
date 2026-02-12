from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I love machine learning",
    "I enjoy studying AI",
    "The pizza was delicious"
]

# Generate embeddings
embeddings = model.encode(sentences)

# Compare similarity
sim_1_2 = cosine_similarity([embeddings[0]], [embeddings[1]])
sim_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])

print("Similarity (1 & 2):", sim_1_2)
print("Similarity (1 & 3):", sim_1_3)