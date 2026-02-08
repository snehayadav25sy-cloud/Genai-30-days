# Day 3 – Tokens & Embeddings (Transformer Fundamentals)

## 📌 Goal of Day 3

Understand **how text is converted into numbers** and how transformers create **embeddings** before attention happens.

By the end of Day 3, you should clearly understand:

* What tokens are
* How tokenization works in practice
* What embeddings represent
* Why embeddings are required before attention

---

## 🧠 Concepts Covered (Simple Language)

### 1. Tokens

Models do not understand raw text. They understand **tokens**, which are small text units.

Example:

```
"I love AI"
→ ["i", "love", "ai"]
```

Special tokens are added:

* `[CLS]` → start of sentence
* `[SEP]` → end of sentence

---

### 2. Token IDs

Each token is mapped to a unique **integer ID** using the model vocabulary.

Example:

```
[CLS] i love ai [SEP]
[101, 1045, 2293, 9932, 102]
```

---

### 3. Embeddings

An **embedding** is a vector (list of numbers) that represents the meaning of a token.

In BERT:

* Each token → **768 numbers**
* These numbers encode meaning + position

Output shape:

```
[batch_size, number_of_tokens, embedding_size]
[1, 5, 768]
```

---

### 4. Why embeddings matter

* They convert discrete IDs into continuous meaning
* Attention operates on embeddings, not raw text
* Contextual understanding begins here

---

## 🧪 Implementation (What we coded)

### File: `embedding_demo.py`

Steps:

1. Load tokenizer (`AutoTokenizer`)
2. Load model (`AutoModel` – encoder only)
3. Tokenize input text
4. Pass tokens through the model
5. Extract embeddings

Successful output:

```
Embedding shape: torch.Size([1, 5, 768])
```

---

## ⚠️ Notes

* `UNEXPECTED` warnings are normal when loading `BertModel`
* They indicate unused task-specific heads
* Safe to ignore for embedding use cases

---

## 🎯 Key Takeaway

> Tokenization converts text into numbers, embeddings convert numbers into meaning, and attention builds understanding from embeddings.

---

## 🔜 Next Day (Day 4 Preview)

* Self-attention intuition
* How tokens attend to each other
* Why transformers beat RNNs

---

✅ Day 3 completed successfully
