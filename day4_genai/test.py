import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ================================
# CONFIG
# ================================
BASE_MODEL = "google/flan-t5-small"
LORA_PATH = "./finetuned_model"   # folder with adapter_model.bin

# ================================
# LOAD TOKENIZER
# ================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ================================
# LOAD BASE MODEL
# ================================
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

# ================================
# LOAD LoRA ADAPTER
# ================================
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()

# ================================
# PROMPT
# ================================
prompt = (
    "Answer the question in 2–3 complete sentences.\n"
    "Question: What is machine learning?\n"
    "Answer:"
)
# ================================
# TOKENIZE
# ================================
inputs = tokenizer(prompt, return_tensors="pt")

# ================================
# GENERATE
# ================================
outputs = model.generate(
    **inputs,
    min_new_tokens=40,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.3
)

# ================================
# OUTPUT
# ================================
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
