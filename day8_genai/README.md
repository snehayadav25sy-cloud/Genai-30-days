# 🚀 Day 8 – Embeddings Deep Dive (GenAI Roadmap)

## 📌 Overview
Day 8 focuses on understanding **text embeddings**, semantic similarity, and how vectors capture meaning in natural language.

Embeddings are the foundation of:
- Semantic Search
- RAG (Retrieval-Augmented Generation)
- Chatbot Memory
- Recommendation Systems
- Clustering & Similarity Matching

---

## 🧠 What I Learned

### 🔹 What Are Embeddings?
Embeddings are dense numerical vector representations of text that capture semantic meaning.

Example:
"I love AI" → `[0.12, -0.98, 0.44, ...]`

Similar meaning → vectors close in space  
Different meaning → vectors far apart  

---

### 🔹 Semantic Similarity
Sentences with similar meaning have high cosine similarity.

Example:
- "I love machine learning"
- "I enjoy studying AI"

These produce high similarity scores (~0.8+).

---

### 🔹 Why Keywords Fail
Traditional keyword search fails because:
- No synonym understanding
- No context awareness
- No paraphrase detection

Example:
Query: "cheap smartphone"
Document: "Affordable mobile phone under budget"

Keyword search → fails  
Embeddings → succeeds  

---

## 🛠 Implementation

### Model Used
`all-MiniLM-L6-v2` from Sentence Transformers

### Installation

```bash
pip install sentence-transformers