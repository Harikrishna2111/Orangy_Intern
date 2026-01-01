import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def load_anand_dataset():
    with open("data/anand_train.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return {"text": lines}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

def main():
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load and tokenize dataset
    raw_dataset = load_anand_dataset()
    dataset = load_dataset("json", data_files={"train": raw_dataset})["train"]
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./anand_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-4,
        warmup_steps=100,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()