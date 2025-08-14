#!/usr/bin/env python3
"""
Embedded Fine-tuning for Ollama
System message gömülü - Her seferinde system message göndermeye gerek yok!
"""

import unsloth
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddedTrainingConfig:
    """Embedded training configuration - System message gömülü"""
    # Model settings
    model_name: str = "unsloth/Phi-4"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training settings
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 20
    max_steps: int = 200
    weight_decay: float = 0.01

    # Dataset settings
    dataset_name: str = "mrkswe/model-04-dataset" # burayı değiştirdin

    # Output settings
    output_dir: str = "./embedded_outputs"
    hub_model_id: str = "mrkswe/llmPhi4Embedded"
    hf_token: str = "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"

    # Embedded behavior settings
    task_prefix: str = "Endpoint seçimi:"
    task_description: str = (
        "Bu soru için en uygun endpoint adını seç. "
        "Sadece endpoint adını yaz. "
        "Hiçbiri uygun değilse 'False' yaz."
    )

    # Advanced settings
    logging_steps: int = 5
    save_steps: int = 25
    eval_steps: int = 25
    early_stopping_patience: int = 5

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class EmbeddedFineTuner:
    """Embedded Fine-tuning - System message davranışı modele gömülü"""

    def __init__(self, config: EmbeddedTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

        # Set random seed
        set_seed(3407)

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def get_chat_template_type(self) -> str:
        """Auto-detect chat template based on model name"""
        model_name_lower = self.config.model_name.lower()

        if "phi" in model_name_lower:
            return "phi-4"
        elif "llama" in model_name_lower:
            return "llama-3"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "gemma" in model_name_lower:
            return "gemma"
        elif "qwen" in model_name_lower:
            return "qwen"
        else:
            return "llama-3"

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info(f"🤖 Loading model: {self.config.model_name}")

        # Load model with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
            loftq_config=None,
        )

        # Setup chat template
        template_type = self.get_chat_template_type()
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=template_type,
        )

        logger.info(f"✅ Model setup completed with {template_type} template")

    def normalize_conversation_format(self, conversations: List[Dict]) -> List[Dict]:
        """Normalize dataset format"""
        normalized = []

        for msg in conversations:
            if "from" in msg and "value" in msg:
                role = "user" if msg["from"] == "human" else "assistant"
                normalized.append({
                    "role": role,
                    "content": msg["value"]
                })
            elif "role" in msg and "content" in msg:
                normalized.append(msg)
            else:
                logger.warning(f"Unknown conversation format: {msg}")

        return normalized

    def create_embedded_training_prompt(self, conversation: List[Dict]) -> str:
        """
        Create training prompts with EMBEDDED behavior
        Model davranışını gömülü olarak öğrenir, system message gereksiz!
        """
        conversation = self.normalize_conversation_format(conversation)

        # Embedded training format
        embedded_conversation = []

        for msg in conversation:
            if msg["role"] == "user":
                # User input'una task behavior'u göm
                enhanced_input = f"{self.config.task_prefix} {msg['content']}"
                embedded_conversation.append({
                    "role": "user",
                    "content": enhanced_input
                })

            elif msg["role"] == "assistant":
                # Assistant output'u direkt endpoint adı olarak tut
                embedded_conversation.append({
                    "role": "assistant",
                    "content": msg["content"]  # Sadece endpoint adı
                })

        return self.tokenizer.apply_chat_template(
            embedded_conversation,
            tokenize=False,
            add_generation_prompt=False
        )

    def formatting_prompts_func(self, examples: Dict) -> Dict:
        """Format examples for training - Fixed multiprocessing issues"""
        conversations = examples["conversations"]
        texts = []

        for conv in conversations:
            try:
                # Normalize conversation format first
                normalized_conv = self.normalize_conversation_format(conv)

                # Create embedded training format
                embedded_conversation = []

                for msg in normalized_conv:
                    if msg["role"] == "user":
                        # Add task prefix to user input
                        enhanced_input = f"{self.config.task_prefix} {msg['content']}"
                        embedded_conversation.append({
                            "role": "user",
                            "content": enhanced_input
                        })

                    elif msg["role"] == "assistant":
                        # Keep assistant response as endpoint name only
                        embedded_conversation.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })

                # Apply chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    embedded_conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )

                texts.append(formatted_text)

            except Exception as e:
                logger.warning(f"Error formatting conversation: {e}")
                # Add empty text for failed conversations
                texts.append("")
                continue

        return {"text": texts}

    def prepare_dataset(self) -> Dataset:
        """Prepare training dataset - Fixed multiprocessing issues"""
        logger.info(f"📊 Loading dataset: {self.config.dataset_name}")

        # Load dataset
        dataset = load_dataset(self.config.dataset_name, split="train")
        logger.info(f"Raw dataset size: {len(dataset)}")

        # Parse JSON conversations - single process to avoid pickle issues
        def parse_conversations(example):
            try:
                example['conversations'] = json.loads(example['conversations'])
                return example
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                example['conversations'] = []
                return example

        # Use single process to avoid multiprocessing pickle issues
        logger.info("Parsing conversations...")
        dataset = dataset.map(parse_conversations, num_proc=1)

        # Filter empty conversations
        logger.info("Filtering empty conversations...")
        dataset = dataset.filter(lambda x: len(x['conversations']) > 0, num_proc=1)

        # Standardize format
        logger.info("Standardizing ShareGPT format...")
        dataset = standardize_sharegpt(dataset)

        # Apply embedded formatting - single process
        logger.info("Applying embedded formatting...")
        dataset = dataset.map(
            self.formatting_prompts_func,
            batched=True,
            num_proc=1,  # Single process to avoid pickle issues
            remove_columns=dataset.column_names,
            desc="Formatting prompts"
        )

        # Filter empty texts
        logger.info("Filtering empty texts...")
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0, num_proc=1)

        logger.info(f"✅ Final dataset size: {len(dataset)} samples")

        # Print a sample to verify formatting
        if len(dataset) > 0:
            logger.info("Sample formatted text:")
            logger.info("-" * 50)
            logger.info(dataset[0]['text'][:500] + "...")
            logger.info("-" * 50)

        return dataset


    # def create_training_arguments(self) -> TrainingArguments:
    #     """Create training arguments - Fixed for newer transformers versions"""
    #     return TrainingArguments(
    #         per_device_train_batch_size=self.config.batch_size,
    #         gradient_accumulation_steps=self.config.gradient_accumulation_steps,
    #         warmup_steps=self.config.warmup_steps,
    #         max_steps=self.config.max_steps,
    #         learning_rate=self.config.learning_rate,
    #         fp16=not is_bfloat16_supported(),
    #         bf16=is_bfloat16_supported(),
    #         logging_steps=self.config.logging_steps,
    #         optim="adamw_8bit",
    #         weight_decay=self.config.weight_decay,
    #         lr_scheduler_type="cosine",
    #         seed=3407,
    #         output_dir=self.config.output_dir,
    #
    #         # Advanced settings - FIXED PARAMETER NAMES
    #         save_strategy="steps",
    #         save_steps=self.config.save_steps,
    #         eval_strategy="steps",  # Changed from 'evaluation_strategy'
    #         eval_steps=self.config.eval_steps,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_loss",
    #         greater_is_better=False,
    #
    #         # Memory optimization
    #         dataloader_pin_memory=False,
    #         remove_unused_columns=False,
    #         gradient_checkpointing=True,
    #         ddp_find_unused_parameters=False,
    #
    #         report_to="none",
    #     )

    def create_training_arguments(self, do_eval: bool = False) -> SFTConfig:
        """Create SFTConfig for trl (compatible with trl==0.21.0).
        do_eval: whether we actually have an eval dataset to run evaluation on.
        """
        return SFTConfig(
            output_dir=self.config.output_dir,
            overwrite_output_dir=False,
            do_train=True,
            do_eval=do_eval,
            eval_strategy="steps" if do_eval else "no",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=max(1, self.config.batch_size),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_steps=self.config.max_steps if self.config.max_steps and self.config.max_steps > 0 else -1,
            warmup_steps=self.config.warmup_steps,
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            save_strategy="steps" if self.config.save_steps else "no",
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if do_eval else None,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            lr_scheduler_type="cosine",
            seed=3407,
            dataset_text_field="text",
            dataset_num_proc=1,
            max_length=self.config.max_seq_length,
            packing=False,
        )


    def train(self):
        """Execute training pipeline"""
        logger.info("🚀 Starting embedded fine-tuning...")

        # Setup model and tokenizer
        self.setup_model_and_tokenizer()

        # Prepare dataset
        train_dataset = self.prepare_dataset()

        # Create eval dataset
        eval_dataset = None
        if len(train_dataset) >= 10:
            split_dataset = train_dataset.train_test_split(test_size=0.1, seed=3407)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Training arguments - pass whether eval dataset exists
        training_args = self.create_training_arguments(do_eval=(eval_dataset is not None))

        # # Setup trainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     processing_class=self.tokenizer,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     dataset_text_field="text",
        #     max_seq_length=self.config.max_seq_length,
        #     dataset_num_proc=1,  # Single process to avoid pickle issues
        #     packing=False,
        #     args=training_args,
        #     callbacks=[
        #         EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)
        #     ] if eval_dataset else None
        # )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,  # doğru: tokenizer -> processing_class
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience)] if eval_dataset else None
        )

        # Start training
        logger.info("📚 Training started...")
        trainer_stats = trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        logger.info(f"✅ Training completed! Loss: {trainer_stats.training_loss:.4f}")
        return trainer_stats

    def export_to_gguf_and_hub(self):
        """Export model to GGUF and HuggingFace Hub"""
        logger.info("📦 Exporting models...")

        quantization_methods = ["q4_k_m", "q8_0", "f16"]

        for quant_method in quantization_methods:
            try:
                logger.info(f"Exporting {quant_method}...")

                self.model.push_to_hub_gguf(
                    repo_id=f"{self.config.hub_model_id}",
                    tokenizer=self.tokenizer,
                    quantization_method=quant_method,
                    token=self.config.hf_token,
                    private=False
                )

                logger.info(f"✅ {quant_method} exported successfully")

            except Exception as e:
                logger.error(f"❌ {quant_method} export failed: {e}")

        # Also push base model
        try:
            self.model.push_to_hub(
                repo_id=self.config.hub_model_id,
                token=self.config.hf_token,
                private=False
            )
            self.tokenizer.push_to_hub(
                repo_id=self.config.hub_model_id,
                token=self.config.hf_token,
                private=False
            )
            logger.info("✅ Base model pushed to Hub")
        except Exception as e:
            logger.error(f"❌ Base model push failed: {e}")

    def create_ollama_usage_files(self):
        """Create Ollama usage files and examples"""

        # 1. Inference example with Python
        python_example = f'''#!/usr/bin/env python3
"""
EMBEDDED MODEL INFERENCE - OLLAMA İÇİN HAZIR!
System message gereksiz - Sadece input gönderin!
"""

from unsloth import FastLanguageModel

def load_embedded_model():
    """Load the embedded fine-tuned model"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        "{self.config.hub_model_id}",  # Your trained model
        max_seq_length={self.config.max_seq_length},
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def predict_endpoint(user_input, model, tokenizer):
    """
    Predict endpoint - NO SYSTEM MESSAGE NEEDED!
    Model davranışı gömülü olarak öğrenildi
    """
    # Same format as training: "Endpoint seçimi: [user_input]"
    formatted_input = f"{self.config.task_prefix} {{user_input}}"

    conversation = [{{"role": "user", "content": formatted_input}}]

    prompt = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,  # Endpoint adı için yeterli
        temperature=0.1,    # Deterministik sonuçlar için
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = result[len(prompt):].strip()

    return prediction

# KULLANIM ÖRNEĞİ
if __name__ == "__main__":
    # Model yükle
    model, tokenizer = load_embedded_model()

    # Test cases
    test_cases = [
        "Üniversite bilgimi eklemek istiyorum. Endpointler: [\\"KullanıcıAdıKaydet\\", \\"UniversiteBilgisiEkle\\", \\"AdresBilgisiGuncelle\\"]",
        "Yeni bir adres ekleyeceğim. Endpointler: [\\"AdresBilgisiGuncelle\\", \\"MeslekBilgisiEkle\\", \\"KullanıcıAdıKaydet\\"]"
    ]

    for test_input in test_cases:
        result = predict_endpoint(test_input, model, tokenizer)
        print(f"Input: {{test_input[:50]}}...")
        print(f"Endpoint: {{result}}")
        print("-" * 50)
'''

        # 2. Ollama Modelfile
        modelfile_content = f'''FROM ./{self.config.hub_model_id.split("/")[-1]}-q4_k_m.gguf

# Template for chat format
TEMPLATE """{{{{ if .System }}}}{{{{ .System }}}}
{{{{ end }}}}{{{{ if .Prompt }}}}### User: {{{{ .Prompt }}}}
{{{{ end }}}}### Assistant: """

# Optimized parameters for endpoint selection
PARAMETER stop "### User:"
PARAMETER stop "### Assistant:"
PARAMETER temperature 0.1
PARAMETER top_k 10
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# System message is NOT needed - model behavior is embedded!
'''

        # 3. Ollama usage guide
        usage_guide = f'''# OLLAMA KULLANIM KILAVUZU

## 1. Model Import
```bash
# GGUF dosyasını indirdikten sonra:
ollama create {self.config.hub_model_id.split("/")[-1]} -f Modelfile
```

## 2. Basic Usage (System Message GEREKSIZ!)
```bash
# Direkt kullanım - format önemli!
ollama run {self.config.hub_model_id.split("/")[-1]} "{self.config.task_prefix} Üniversite bilgimi eklemek istiyorum. Endpointler: [\\"KullanıcıAdıKaydet\\", \\"UniversiteBilgisiEkle\\", \\"AdresBilgisiGuncelle\\"]"

# Output: UniversiteBilgisiEkle
```

## 3. API Usage
```bash
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{self.config.hub_model_id.split("/")[-1]}",
    "prompt": "{self.config.task_prefix} Your endpoint selection question here...",
    "stream": false
  }}'
```

## 4. Python API
```python
import requests

def ask_ollama(question):
    response = requests.post("http://localhost:11434/api/generate", json={{
        "model": "{self.config.hub_model_id.split("/")[-1]}",
        "prompt": f"{self.config.task_prefix} {{question}}",
        "stream": False
    }})
    return response.json()["response"].strip()

# Usage
result = ask_ollama("Üniversite bilgimi eklemek istiyorum. Endpointler: [...]")
print(result)  # Output: UniversiteBilgisiEkle
```

## ÖNEMLİ NOTLAR:
- ✅ System message GEREKSIZ - model davranışı gömülü!
- ✅ Format önemli: "{self.config.task_prefix} [your_question]"
- ✅ Model sadece endpoint adı döndürür
- ✅ Hiçbiri uygun değilse "False" döndürür
'''

        # Save all files
        files = [
            ("inference_example.py", python_example),
            ("Modelfile", modelfile_content),
            ("OLLAMA_USAGE.md", usage_guide)
        ]

        for filename, content in files:
            filepath = Path(self.config.output_dir) / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"📝 Created: {filepath}")

    def run_complete_pipeline(self):
        """Run complete embedded training pipeline"""
        try:
            logger.info("🎯 Starting EMBEDDED fine-tuning for Ollama...")

            # Train model
            trainer_stats = self.train()

            # Export models
            self.export_to_gguf_and_hub()

            # Create Ollama files
            self.create_ollama_usage_files()

            logger.info("🎉 EMBEDDED training completed successfully!")
            logger.info("🔥 Model ready for Ollama - NO SYSTEM MESSAGE NEEDED!")
            return trainer_stats

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


def main():
    """Main execution with embedded configuration"""

    # 🤖 AVAILABLE MODELS
    available_models = {
        "phi-4": "unsloth/Phi-4",
        "phi-4-4bit": "unsloth/Phi-4-unsloth-bnb-4bit",
        "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "mistral-small": "unsloth/Mistral-Small-Instruct-2409",
        "gemma-2-9b": "unsloth/gemma-2-9b-bnb-4bit",
        "qwen-2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    }

    # 📝 MODEL SELECTION
    SELECTED_MODEL = "llama-3.2-3b"  # Change this to select different model

    # ⚙️ EMBEDDED CONFIGURATION
    config = EmbeddedTrainingConfig(
        # Model settings
        model_name=available_models[SELECTED_MODEL],
        max_seq_length=2048,
        load_in_4bit=True,

        # LoRA settings
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.05,

        # Training settings
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=20,
        max_steps=200,  # Increase for better performance
        weight_decay=0.01,

        # Dataset
        dataset_name="mrkswe/model-04-dataset",# burayı değiştirdim

        # Output settings
        output_dir="./embedded_outputs",
        hub_model_id="mrkswe/phi4-model04",  # Your model name
        hf_token="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",

        # Embedded behavior - BU DAVRANIŞI GÖMER!
        task_prefix="Endpoint seçimi:",
        task_description="En uygun endpoint adını seç. Sadece endpoint adını yaz.",

        # Advanced settings
        logging_steps=5,
        save_steps=25,
        eval_steps=25,
        early_stopping_patience=5
    )

    print("🎯 EMBEDDED FINE-TUNING FOR OLLAMA")
    print("=" * 50)
    print(f"🤖 Model: {SELECTED_MODEL} ({config.model_name})")
    print(f"📊 Dataset: {config.dataset_name}")
    print(f"💾 Output: {config.output_dir}")
    print(f"🔗 Hub: {config.hub_model_id}")
    print(f"🎯 Task Prefix: '{config.task_prefix}'")
    print(f"🔥 Ollama Ready: YES - No system message needed!")
    print("=" * 50)

    # Run embedded fine-tuning
    fine_tuner = EmbeddedFineTuner(config)
    trainer_stats = fine_tuner.run_complete_pipeline()

    print("\\n🎉 EMBEDDED TRAINING COMPLETED!")
    print(f"📊 Final Loss: {trainer_stats.training_loss:.4f}")
    print(f"📦 Check '{config.output_dir}/' for:")
    print("   - inference_example.py (Python usage)")
    print("   - Modelfile (Ollama import)")
    print("   - OLLAMA_USAGE.md (Complete guide)")
    print("\\n🔥 OLLAMA USAGE:")
    print(
        f'   ollama run {config.hub_model_id.split("/")[-1]} \\ '
        f'{config.task_prefix} Your question here\\'
    )
    print("\n✅ NO SYSTEM MESSAGE NEEDED!")


if __name__ == "__main__":

    main()







