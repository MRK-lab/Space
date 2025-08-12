import json
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from unsloth import FastLanguageModel

MAX_LEN = 128
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" 
 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LEN,
    dtype=torch.float16,
    load_in_4bit=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

#lora
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

PROMPT_TEMPLATE = """Aşağıda bir istek JSON formatında 'userMessage' yer alıyor.
Bu isteğe bağlı olarak 'required=true' olan kısımları uygun şekilde doldur,
eksiklikleri kullanıcıdan iste.

Soru: {question}
Cevap:{answer}"""

dataset_list = [
    {"instruction": "Verilen müşteri bilgilerini kullanarak yeni bir müşteri kaydı oluştur.",
     "input": {"Name": "Ahmet Yılmaz", "Email": "ahmet.yilmaz@example.com", "Phone": "+90 532 111 22 33"},
     "output": {"status": True, "message": "Müşteri başarıyla oluşturuldu.", "customer_id": 101}},
    {"instruction": "Sipariş bilgilerini kullanarak yeni bir sipariş kaydı oluştur.",
     "input": {"OrderID": None, "CustomerID": 101, "Product": "Laptop", "Quantity": 2},
     "output": {"status": True, "message": "Sipariş başarıyla oluşturuldu.", "OrderID": 202}},
    {"instruction": "Ürün stok miktarını güncelle.",
     "input": {"ProductID": 55, "NewStock": 120},
     "output": {"status": True, "message": "Stok miktarı güncellendi."}},
    {"instruction": "Müşteri ID'si verilen müşterinin bilgilerini getir.",
     "input": {"CustomerID": 101},
     "output": {"status": True, "Name": "Ahmet Yılmaz", "Email": "ahmet.yilmaz@example.com", "Phone": "+90 532 111 22 33"}},
    {"instruction": "Belirtilen siparişi iptal et.",
     "input": {"OrderID": 202},
     "output": {"status": True, "message": "Sipariş iptal edildi."}},
    {"instruction": "Tüm aktif müşterileri listele.",
     "input": {},
     "output": {"status": True, "customers": [{"CustomerID": 101, "Name": "Ahmet Yılmaz"}, {"CustomerID": 102, "Name": "Mehmet Demir"}]}}
]

dataset = Dataset.from_list(dataset_list)

def preprocess_function(examples):
    inputs = []
    for inst, inp in zip(examples["instruction"], examples["input"]):
        inp_json = json.dumps(inp, ensure_ascii=False)
        prompt = PROMPT_TEMPLATE.format(question=f"Instruction: {inst}\nInput: {inp_json}\n", answer=f"Output:\n")
        inputs.append(prompt)

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )

    outputs = [json.dumps(out, ensure_ascii=False) for out in examples["output"]]
    tokenized_labels = tokenizer(
        outputs,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )["input_ids"]

  
    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in lbl] for lbl in tokenized_labels]

    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=MAX_LEN)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    report_to=[],  # W&B kapalı
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

from huggingface_hub import login
login("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") 
#trainer.push_to_hub("01Nur/06_fine")
trainer.push_to_hub_gguf("01Nur/06_fine", tokenizer, quantization_method = "q4_k_m", token = "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")


