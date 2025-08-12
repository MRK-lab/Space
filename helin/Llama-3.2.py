from multiprocessing import freeze_support
import json
import torch

from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from trl import SFTTrainer, SFTConfig
from multiprocessing import freeze_support


from unsloth.chat_templates import standardize_sharegpt, get_chat_template


def main():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True


    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"  



    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token="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"
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


    dataset = load_dataset("mrkswe/llmEndpointDatasetConversation_2", split="train")


    def parse_conversations(example):
        example['conversations'] = json.loads(example['conversations'])
        return example

    dataset = dataset.map(parse_conversations)


    dataset = standardize_sharegpt(dataset)


    tokenizer = get_chat_template(tokenizer, chat_template="Llama-3.1-8B")


    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}


    dataset = dataset.map(formatting_prompts_func, batched=True)


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()


    token = "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"
    create_repo("helin-lora-llama-3.1", repo_type="model", private=False, token=token, exist_ok=True)



if __name__ == "__main__":
    freeze_support()
    main()