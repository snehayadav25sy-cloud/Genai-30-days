from transformers import pipeline

analysis = pipeline("sentiment-analysis",
                     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english" )

texts=[
    "I love chat-gpt ",
    "i have been waiting for the hugging face course"
]

results = analysis(texts)

for text,result in zip(texts,results):
    print(f"Text:{text}")
    print(f"sentiment:{result['label']} | score:{result['score']:.4f}")
    print("="*40)