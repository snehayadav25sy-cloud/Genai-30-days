# Day 4 – Lightweight Fine-Tuning with LoRA (PEFT)

This project demonstrates **lightweight fine-tuning of a Large Language Model using LoRA (Low-Rank Adaptation)** on a very small custom dataset. The goal is to understand **fine-tuning vs prompting**, overfitting risks, and how **PEFT (Parameter-Efficient Fine-Tuning)** works in practice.

---

## 🚀 What This Project Covers

* What fine-tuning is and **when NOT to do it**
* Difference between **prompting vs fine-tuning**
* Overfitting risks with small datasets
* **LoRA (Low-Rank Adaptation)** – a PEFT technique
* Fine-tuning a model with **20–50 samples**
* Running inference using a fine-tuned adapter

---

## 🧠 Technique Used

**LoRA (Low-Rank Adaptation)**

* Category: **PEFT (Parameter-Efficient Fine-Tuning)**
* Base model weights are frozen
* Only small adapter layers are trained
* Very low compute and memory cost
* Industry-standard approach for LLM customization

---

## 🏗️ Model & Tools

* **Base Model:** `google/flan-t5-small`
* **Framework:** Hugging Face Transformers
* **PEFT Library:** `peft`
* **Language:** Python

---

## 📂 Project Structure

```
day4_genai/
│── data.json                # Small custom training dataset
│── train.py                 # LoRA fine-tuning script
│── test.py                  # Inference / testing script
│── finetuned_model/
│   └── checkpoint-45/       # Saved LoRA adapter
│       ├── adapter_config.json
│       └── adapter_model.safetensors
```

---

## 📊 Dataset Example

```json
{
  "input": "Hello",
  "output": "Hi! How can I help you today?"
}
```

> ⚠️ Note: The dataset is intentionally small to demonstrate **overfitting and behavior changes** during fine-tuning.

---

## 🏃 How to Run

### 1️⃣ Create Virtual Environment

```bash
python -m venv mygenaivenv
mygenaivenv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install transformers datasets peft accelerate torch
```

### 3️⃣ Train (LoRA Fine-Tuning)

```bash
python train.py
```

### 4️⃣ Test the Fine-Tuned Model

```bash
python test.py
```

---

## 🧪 Observations

* With very small datasets, the model may:

  * Repeat tokens
  * Give short or conservative outputs
* This behavior demonstrates **overfitting**, which is expected
* Despite this, LoRA successfully influences model behavior

---

## 📌 Key Learnings

* Fine-tuning is **expensive** compared to prompting
* LoRA enables cheap, fast customization
* Small datasets = high overfitting risk
* Always match the **same base model** during training and inference

---

## 📚 When to Use Fine-Tuning

✅ Domain-specific language
✅ Consistent response style
✅ Repeated task patterns

❌ Simple Q&A
❌ One-off tasks
❌ Small improvements achievable via prompting

---

## 🔮 Next Improvements

* Add more diverse training samples
* Compare **base model vs LoRA outputs**
* Try Prompt Tuning / Prefix Tuning
* Increase LoRA rank (`r` value)
* Log training loss and evaluation metrics

---

## 🏁 Conclusion

This project provides a **hands-on, minimal example** of lightweight fine-tuning using LoRA. It is ideal for learning PEFT concepts and understanding the trade-offs between prompting and fine-tuning.

---

👩‍💻 *Built as part of a GenAI learning journey*
