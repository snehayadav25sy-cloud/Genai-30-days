import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def summarize_text(text):
    response=client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role":"system","content":"you are an expert text summarizer."},
            {"role":"user","content":f"summarize this text:\n{text}"}
        
                 ],
        temperature=0.3         
    )
    return response.choices[0].message.content  

def answer_question(context , question):
    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Answer only from the given context."},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion:{question}"}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def generate_content(prompt):
    response=client.chat.completions.create(
       model="gpt-4o-mini", 
       messages=[
           {"role":"system","content":"you are a creative content writer."},
           {"role":"user","content":prompt}
       ],
       temperature=0.8
    )
    return response.choices[0].message.content
