#     !pip install unsloth
#      sudo apt update
#       sudo apt install libcurl4-openssl-dev

import os
import json
import subprocess

from datasets import load_dataset, DownloadConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, PrefixTuningConfig, get_peft_model
from trl import SFTTrainer
from unsloth.chat_templates import standardize_sharegpt



HF_TOKEN = ""
DATASET_ID = "mrkswe/llmEndpointDatasetConversation_2"
OLLAMA_MODEL = ""

SYSTEM_MSG = (
    "Yalnızca uygun endpoint adını döndür. "
    "Hiçbiri uygun değilse veya format bozuksa False yaz."
)


BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./outputs"



def parse_conversations(ex):
    ex["conversations"] = json.loads(ex["conversations"])
    return ex


def formatting_prompts_func(examples):
    # system/user/assistant prompt’larını tek string’e dönüştür
    texts = []
    for conv in examples["conversations"]:
        prompt = f"[SYSTEM] {SYSTEM_MSG}\n"
        for msg in conv:
            role = "[USER]" if msg["role"] == "user" else "[ASSISTANT]"
            prompt += f"{role} {msg['content']}\n"
        texts.append(prompt)
    return {"text": texts}


def prepare_dataset() -> "Dataset":
    dl_cfg = DownloadConfig(max_retries=10)
    ds = load_dataset(
        DATASET_ID,
        split="train",
        num_proc=1,
        download_config=dl_cfg,
    )
    ds = ds.map(parse_conversations, num_proc=1)
    ds = standardize_sharegpt(ds) #unsloth burada kullanılıyor sadece
    ds = ds.map(formatting_prompts_func, batched=True, num_proc=1)
    return ds



def prepare_model_and_tokenizer():
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_use_double_quant  = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_conf,
        device_map="auto",
    )
    return model, tokenizer



def apply_peft(model):
    # A) LoRA ayarları
    lora_cfg = LoraConfig(
        r              = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha     = 16,
        lora_dropout   = 0.0,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)

    # B) Prefix-Tuning ayarları
    prefix_cfg = PrefixTuningConfig(
        task_type           = "CAUSAL_LM",
        num_virtual_tokens  = 20,
        inference_mode      = False,
        encoder_hidden_size = model.config.hidden_size,
    )
    model = get_peft_model(model, prefix_cfg)
    return model




def train_model(model, tokenizer, dataset):
    tokenizer.model_max_length = 2048
    trainer = SFTTrainer(
        model              = model,
        # tokenizer          = tokenizer,
        train_dataset      = dataset,
        # dataset_text_field = "text",
        # max_seq_length     = 2048,
        data_collator      = DataCollatorForSeq2Seq(tokenizer),
        args = TrainingArguments(
            per_device_train_batch_size   = 1,
            gradient_accumulation_steps   = 4,
            warmup_steps                  = 5,
            max_steps                     = 1000,
            learning_rate                 = 2e-4,
            fp16                          = True,
            logging_steps                 = 50,
            output_dir                    = OUTPUT_DIR,
            report_to                     = "none",
        ),
    )
    trainer.train()



def export_to_gguf_and_ollama(model, tokenizer):
    """
    Modeli önce disk altına HF formatında kaydeder,
    sonra GGUF'ye dönüştürüp HuggingFace Hub'a push eder.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # HF formatında kaydet
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # GGUF formatında push (ollama ile değil HF Hub üzerinden)
    model.push_to_hub_gguf(
        repo_id=OLLAMA_MODEL,
        tokenizer=tokenizer,
        quantization_method="q4_k_m",
        token=HF_TOKEN
    )
    print(f"✅ GGUF model Hub’a yüklendi: {OLLAMA_MODEL}")




def main():
    ds = prepare_dataset()
    model, tokenizer = prepare_model_and_tokenizer()
    model = apply_peft(model)
    train_model(model, tokenizer, ds)
    export_to_gguf_and_ollama(model, tokenizer)

if __name__ == "__main__":
    main()

