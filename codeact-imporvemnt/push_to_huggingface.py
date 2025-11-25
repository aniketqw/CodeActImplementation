"""
Push trained CodeAct model to Hugging Face Hub
"""
import os
import json
import shutil
from pathlib import Path

def create_model_card():
    """Create README.md for the model"""
    return """---
license: apache-2.0
language:
- en
tags:
- code
- codeact
- python
- mlx
- lora
base_model: Qwen/Qwen2.5-3B
pipeline_tag: text-generation
---

# CodeAct Fine-tuned Qwen2.5-3B

A fine-tuned version of Qwen2.5-3B for code generation with self-evaluation feedback.

## Model Description

This model was fine-tuned using the CodeAct approach with:
- **Base Model:** Qwen/Qwen2.5-3B
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Training Data:** 100 curated Python programming examples
- **Categories:** Math, Strings, Lists, Algorithms, Data Structures

## Usage

### With MLX (Apple Silicon)
```python
from mlx_lm import load, generate

model, tokenizer = load("YOUR_USERNAME/codeact-qwen2.5-3b")
# Or with adapter:
# model, tokenizer = load("Qwen/Qwen2.5-3B", adapter_path="YOUR_USERNAME/codeact-qwen2.5-3b")

response = generate(model, tokenizer, prompt="Calculate factorial of 5", max_tokens=200)
print(response)
```

### With PyTorch (CUDA/CPU)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/codeact-qwen2.5-3b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
```

### Interactive Demo
```bash
# Auto-detect backend (MLX/CUDA/CPU)
python interactive_universal.py

# Force specific backend
python interactive_universal.py --backend cuda
python interactive_universal.py --backend mlx
python interactive_universal.py --backend cpu
```

## Training Details

- **Iterations:** 500
- **Batch Size:** 1
- **LoRA Layers:** 16
- **Learning Rate:** 1e-5
- **Platform:** Apple M3 (MLX)

## Response Format

The model uses structured tags:
- `<thought>reasoning</thought>` - Chain of thought
- `<execute>code</execute>` - Python code to execute
- `<solution>answer</solution>` - Final answer
- `<feedback>assessment</feedback>` - Self-evaluation

## Example

**Input:** "Calculate the sum of squares from 1 to 10"

**Output:**
```
<thought>Sum of squares formula: n(n+1)(2n+1)/6</thought>

<execute>
n = 10
result = n * (n + 1) * (2 * n + 1) // 6
print(result)
</execute>

<solution>Sum of squares from 1 to 10 is 385</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Used O(1) formula instead of O(n) loop
</feedback>
```

## License

Apache 2.0
"""

def push_to_hub(repo_name, model_dir="./models/codeact-mlx-qwen2.5-3b"):
    """Push model to Hugging Face Hub"""
    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi()

    # Check if logged in
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Logged in as: {username}")
    except Exception as e:
        print("Not logged in to Hugging Face!")
        print("Run: huggingface-cli login")
        return

    # Full repo name
    full_repo_name = f"{username}/{repo_name}"
    print(f"\nPushing to: {full_repo_name}")

    # Create temp directory with all files
    temp_dir = "./temp_upload"
    os.makedirs(temp_dir, exist_ok=True)

    # Copy model files
    for file in os.listdir(model_dir):
        src = os.path.join(model_dir, file)
        dst = os.path.join(temp_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {file}")

    # Create model card
    readme_path = os.path.join(temp_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(create_model_card().replace("YOUR_USERNAME", username))
    print("  Created: README.md")

    # Copy interactive script
    script_src = "./interactive_universal.py"
    if os.path.exists(script_src):
        shutil.copy2(script_src, os.path.join(temp_dir, "interactive_universal.py"))
        print("  Copied: interactive_universal.py")

    # Create or get repo
    try:
        create_repo(full_repo_name, exist_ok=True)
        print(f"\nRepository ready: https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"Repo exists or error: {e}")

    # Upload
    print("\nUploading files...")
    try:
        upload_folder(
            folder_path=temp_dir,
            repo_id=full_repo_name,
            commit_message="Upload CodeAct fine-tuned model"
        )
        print(f"\nSuccess! Model available at:")
        print(f"  https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"Upload error: {e}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nCleanup complete.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Push model to Hugging Face")
    parser.add_argument("--repo", type=str, default="codeact-qwen2.5-3b",
                       help="Repository name on HuggingFace")
    parser.add_argument("--model-dir", type=str, default="./models/codeact-mlx-qwen2.5-3b",
                       help="Local model directory")

    args = parser.parse_args()

    print("="*60)
    print("Push CodeAct Model to Hugging Face")
    print("="*60)

    push_to_hub(args.repo, args.model_dir)

if __name__ == "__main__":
    main()
