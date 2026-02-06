# Day 2 – Transformer Architecture (Big Picture)

## 🎯 Goal
Understand how transformers work at a conceptual level and see the practical difference between encoder and decoder models.

No math. No equations. Just flow and intuition.

---

## 🧠 Concepts Covered

### 1. How text flows through a transformer
- Text → Tokens
- Tokens → Embeddings
- Positional Encoding
- Self-Attention
- Output layer

---

### 2. Why Attention is the breakthrough
- Replaced RNNs and LSTMs
- Allows every word to attend to every other word
- Captures long-range dependencies
- Enables parallel processing

---

### 3. Encoder vs Decoder

| Component | Purpose | Example |
|--------|--------|--------|
| Encoder | Understand input | BERT |
| Decoder | Generate text | GPT |

---

### 4. Why BERT ≠ GPT

| Feature | BERT | GPT |
|------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Bidirectional | Yes | No |
| Text generation | ❌ | ✅ |
| Best for | Understanding | Generation |

---

### 5. Pretrained model behavior
Same input, different behavior:
- BERT → classification / analysis
- GPT → continuation / generation

---

