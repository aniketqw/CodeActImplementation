"""
Fine-tune with MLX on Apple Silicon (M1/M2/M3)
Uses MLX-LM for efficient training on Mac GPUs.
"""
import os
import json
import subprocess
from pathlib import Path

def prepare_mlx_data():
    """Convert training data to MLX format"""

    print("\n" + "="*60)
    print("Preparing Data for MLX Training")
    print("="*60)

    # Read the original training data (100 examples)
    input_file = "data/codeact_feedback_train_100.jsonl"
    output_dir = "data/mlx_train"

    os.makedirs(output_dir, exist_ok=True)

    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    # Convert to MLX chat format
    mlx_data = []
    for ex in examples:
        mlx_data.append({"messages": ex["messages"]})

    # Split into train and valid (90/10)
    split_idx = max(1, int(len(mlx_data) * 0.9))
    train_data = mlx_data[:split_idx]
    valid_data = mlx_data[split_idx:] if split_idx < len(mlx_data) else mlx_data[-1:]

    # Write train.jsonl
    train_file = os.path.join(output_dir, "train.jsonl")
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Write valid.jsonl
    valid_file = os.path.join(output_dir, "valid.jsonl")
    with open(valid_file, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Training examples: {len(train_data)}")
    print(f"✓ Validation examples: {len(valid_data)}")
    print(f"✓ Data saved to: {output_dir}/")

    return output_dir

def finetune_with_mlx():
    """Fine-tune using MLX-LM on Apple Silicon"""

    print("="*60)
    print("Fine-tuning with MLX on Apple Silicon")
    print("="*60)

    # Check MLX availability
    try:
        import mlx.core as mx
        print(f"✓ MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'installed'}")
        print(f"✓ Using Apple Metal GPU acceleration")
    except ImportError:
        print("✗ MLX not available. Install with: pip install mlx mlx-lm")
        return

    # Prepare data
    data_dir = prepare_mlx_data()

    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)

    model_name = "Qwen/Qwen2.5-3B"
    output_dir = "./models/codeact-mlx-qwen2.5-3b"

    print(f"✓ Base model: {model_name}")
    print(f"✓ Output dir: {output_dir}")
    print(f"✓ Method: LoRA (Low-Rank Adaptation)")

    print("\n" + "="*60)
    print("Starting MLX Training")
    print("="*60 + "\n")

    # Use mlx_lm lora to fine-tune
    # Increased iters for 100 examples, more layers for 3B model
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--adapter-path", output_dir,
        "--iters", "500",
        "--batch-size", "1",
        "--num-layers", "16",
        "--learning-rate", "1e-5",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode == 0:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"✓ LoRA adapter saved to: {output_dir}")
        print("\nTo use the fine-tuned model:")
        print(f'  python -m mlx_lm.generate --model {model_name} --adapter-path {output_dir} --prompt "Your prompt here"')
    else:
        print(f"\n✗ Training failed with exit code: {result.returncode}")

    return output_dir

if __name__ == "__main__":
    finetune_with_mlx()
