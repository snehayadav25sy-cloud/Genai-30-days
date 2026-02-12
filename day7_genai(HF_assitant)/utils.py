import torch
from transformers import pipeline

# Detect device (0 = GPU, -1 = CPU)
device = 0 if torch.cuda.is_available() else -1

# Load models once (VERY IMPORTANT - prevents reloading every time)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=device
)

generator = pipeline(
    "text-generation",
    model="gpt2",
    device=device
)


# ----------- SUMMARIZATION -----------

def summarize_text(text):
    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        do_sample=False,
        truncation=True
    )
    return result[0]["summary_text"]


# ----------- QUESTION ANSWERING -----------

def answer_question(context, question):
    prompt = f"""
    Answer the question based only on the context below.

    Context: {context}
    Question: {question}
    """

    result = qa_pipeline(
        prompt,
        max_length=150,
        do_sample=False
    )
    return result[0]["generated_text"]


# ----------- CONTENT GENERATION -----------

def generate_content(prompt):
    result = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return result[0]["generated_text"]