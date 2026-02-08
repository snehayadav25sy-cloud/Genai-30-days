from transformers import pipeline

generator=pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence will change the world because"

output = generator(prompt,max_length=50,do_sample=True,temperature=0.7,top_p=0.9)

print(output[0]["generated_text"])