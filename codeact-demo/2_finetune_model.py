"""
Step 2: Fine-tune Qwen2-0.5B on CodeAct Dataset
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
import warnings
warnings.filterwarnings('ignore')

def finetune_model():
    """Fine-tune the model"""

    print("\n" + "="*60)
    print("Step 1: Loading Dataset")
    print("="*60)

    # Load dataset
    dataset = load_dataset('json', data_files='data/codeact_train.jsonl', split='train')
    print(f"✓ Loaded {len(dataset)} training examples")

    print("\n" + "="*60)
    print("Step 2: Loading Model and Tokenizer")
    print("="*60)

    model_name = "Qwen/Qwen2-0.5B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Loaded tokenizer")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    print(f"✓ Loaded model")
    print(f"  Parameters: {model.num_parameters() / 1e6:.2f}M")

    print("\n" + "="*60)
    print("Step 3: Preparing Dataset")
    print("="*60)

    # Format conversations
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    formatted_dataset = dataset.map(format_chat)
    print(f"✓ Formatted conversations")

    # Tokenize
    def tokenize(examples):
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = formatted_dataset.map(
        tokenize,
        remove_columns=formatted_dataset.column_names,
        batched=True
    )

    print(f"✓ Tokenized dataset")
    print(f"  Average tokens per example: {sum(len(x['input_ids']) for x in tokenized_dataset) / len(tokenized_dataset):.0f}")

    print("\n" + "="*60)
    print("Step 4: Setting Up Training")
    print("="*60)

    # Training arguments
    output_dir = "./models/codeact-qwen2-0.5b"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("✓ Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output dir: {output_dir}")

    print("\n" + "="*60)
    print("Step 5: Training Model")
    print("="*60)
    print("This will take a few minutes...\n")

    # Train
    trainer.train()

    print("\n" + "="*60)
    print("Step 6: Saving Model")
    print("="*60)

    # Save final model
    final_dir = "./models/codeact-qwen2-0.5b-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"✓ Model saved to: {final_dir}")
    print(f"✓ Training complete!")

    return final_dir

if __name__ == "__main__":
    print("="*60)
    print("Fine-tuning CodeAct Model")
    print("="*60)
    finetune_model()
