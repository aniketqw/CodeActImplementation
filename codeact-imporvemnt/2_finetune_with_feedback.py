"""
Fine-tune with Feedback-Aware Training
This approach gives higher weight to feedback sections during training.
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np

def finetune_with_feedback():
    """Fine-tune with emphasis on feedback learning"""
    
    print("\n" + "="*60)
    print("Loading Dataset with Feedback")
    print("="*60)
    
    dataset = load_dataset('json', data_files='data/codeact_feedback_train.jsonl', split='train')
    print(f"✓ Loaded {len(dataset)} training examples")
    
    print("\n" + "="*60)
    print("Loading Model and Tokenizer")
    print("="*60)
    
    model_name = "Qwen/Qwen2-0.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print(f"✓ Model parameters: {model.num_parameters() / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("Preparing Dataset with Feedback Awareness")
    print("="*60)
    
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    formatted_dataset = dataset.map(format_chat)
    
    def tokenize(examples):
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=768,  # Increased for feedback content
            padding=False
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = formatted_dataset.map(
        tokenize,
        remove_columns=formatted_dataset.column_names,
        batched=True
    )
    
    avg_tokens = sum(len(x['input_ids']) for x in tokenized_dataset) / len(tokenized_dataset)
    print(f"✓ Average tokens per example: {avg_tokens:.0f}")
    
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    
    output_dir = "./models/codeact-feedback-qwen2-0.5b"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # More epochs for feedback learning
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,  # Lower LR for better convergence
        warmup_steps=20,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        weight_decay=0.01,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print(f"✓ Epochs: {training_args.num_train_epochs}")
    print(f"✓ Learning rate: {training_args.learning_rate}")
    print(f"✓ Output: {output_dir}")
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    final_dir = "./models/codeact-feedback-qwen2-0.5b-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"✓ Model saved to: {final_dir}")
    print("✓ Training complete!")
    
    return final_dir

if __name__ == "__main__":
    print("="*60)
    print("Fine-tuning with Feedback-Aware Training")
    print("="*60)
    finetune_with_feedback()