
from transformers import pipeline

analysis= pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
 
texts=["i love ai ",
      "i have been waiting for the hugging face course",
      "i love the course more than anything "
      ]

results = analysis(texts)

#print(result)

for text,result in zip(texts,results):
    print(f'Text : {text}')
    print(f'sentiment: {result['label']} | Score : {result['score']:.4f}')
    print("-" * 40)


    