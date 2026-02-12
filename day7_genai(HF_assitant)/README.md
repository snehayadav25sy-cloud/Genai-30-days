GenAI Day 7 – Hugging Face AI Text Assistant
📌 Project Overview

This project is a local AI Text Assistant built using Hugging Face Transformers.

It supports:

📝 Text Summarization

❓ Question Answering (Context-based)

✍️ Creative Content Generation

All models run locally on CPU or GPU.

🛠 Technologies Used

Python

PyTorch

Hugging Face Transformers

Streamlit (for UI)

CUDA (optional for GPU acceleration)

🤖 Models Used
Task	Model
Summarization	facebook/bart-large-cnn
Question Answering	google/flan-t5-base
Text Generation	gpt2
📂 Project Structure
day6_genai/
│
├── app.py
├── utils.py
├── requirements.txt
└── README.md
⚙️ Installation
1️⃣ Create virtual environment
python -m venv mygenaivenv
mygenaivenv\Scripts\activate
2️⃣ Install dependencies
pip install -r requirements.txt
▶️ Run the App
streamlit run app.py
🧠 How It Works

Models are loaded once using Hugging Face pipeline.

GPU is automatically detected.

Each function calls its respective model.

Results are returned and displayed via Streamlit UI.

🚀 Future Improvements

Add RAG with FAISS

Add model selection dropdown

Add conversation memory

Add quantized models for low RAM systems

🏆 What You Learned in Day 7

Using Hugging Face pipelines

Running LLM locally

Device management (CPU vs GPU)

Controlling generation parameters

Building modular AI utilities