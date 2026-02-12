import streamlit as st
from utils import summarize_text, answer_question, generate_content

st.set_page_config(page_title="AI Text Assistant (HF)", layout="centered")

st.title("🤖 AI Text Assistant (HuggingFace Version)")
st.markdown("Built with Open Source Models")

option = st.sidebar.selectbox(
    "Choose Feature",
    ("Summarize Text", "Ask Question", "Generate Content")
)

# -----------------------------
if option == "Summarize Text":
    st.subheader("📝 Text Summarization")
    text = st.text_area("Enter text")

    if st.button("Summarize"):
        if text.strip():
            with st.spinner("Summarizing..."):
                result = summarize_text(text)
            st.success("Summary:")
            st.write(result)

# -----------------------------
elif option == "Ask Question":
    st.subheader("❓ Context-Based Q&A")
    context = st.text_area("Enter context")
    question = st.text_input("Enter question")

    if st.button("Get Answer"):
        if context.strip() and question.strip():
            with st.spinner("Thinking..."):
                result = answer_question(context, question)
            st.success("Answer:")
            st.write(result)

# -----------------------------
elif option == "Generate Content":
    st.subheader("✍️ Content Generator")
    prompt = st.text_area("Enter prompt")

    if st.button("Generate"):
        if prompt.strip():
            with st.spinner("Generating..."):
                result = generate_content(prompt)
            st.success("Generated Content:")
            st.write(result)