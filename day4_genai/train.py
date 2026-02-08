from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

# 1️⃣ Load dataset
dataset = load_dataset("json", data_files="data.json")

# 2️⃣ Load base model + tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3️⃣ Apply LoRA (PEFT)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],   # T5 attention projections
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4️⃣ Tokenization
def preprocess(example):
    inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    outputs = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# 5️⃣ Training config
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# 7️⃣ Train
trainer.train()

# 8️⃣ Save LoRA adapter + tokenizer
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")