import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

model_id = "microsoft/Phi-4-mini-flash-reasoning"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

df = pd.read_csv("dataset_06.csv")  
print(df.head())
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize etme işlemi
tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy='epochs',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)
trainer.train()
# token: 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
model.push_to_hub("BeyzaNurYldrmm/06_fine")
tokenizer.push_to_hub("BeyzaNurYldrmm/06_fine")


