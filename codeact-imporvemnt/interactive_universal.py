"""
Universal CodeAct Interactive Demo
Supports: CUDA (NVIDIA), MLX (Apple Silicon), CPU
Auto-detects best available backend
"""
import re
import sys
import os
import argparse
from io import StringIO

# ============= BACKEND DETECTION =============
def detect_backend():
    """Auto-detect the best available backend"""
    # Check for MLX (Apple Silicon)
    try:
        import mlx.core as mx
        return "mlx"
    except ImportError:
        pass

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Check for MPS (Apple Metal via PyTorch)
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass

    # Fallback to CPU
    return "cpu"

# ============= MLX BACKEND =============
class MLXBackend:
    def __init__(self, model_name, adapter_path=None):
        from mlx_lm import load, generate
        self.generate_fn = generate

        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading MLX model with adapter: {adapter_path}")
            self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        else:
            print(f"Loading MLX model: {model_name}")
            self.model, self.tokenizer = load(model_name)

    def generate(self, prompt, max_tokens=400):
        return self.generate_fn(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )

# ============= PYTORCH BACKEND (CUDA/MPS/CPU) =============
class PyTorchBackend:
    def __init__(self, model_name, device="auto", adapter_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading PyTorch model on {self.device}: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model with appropriate dtype
        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        # Load LoRA adapter if available
        if adapter_path and os.path.exists(adapter_path):
            try:
                from peft import PeftModel
                print(f"Loading LoRA adapter: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                print("Warning: peft not installed, skipping adapter")

    def generate(self, prompt, max_tokens=400):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        return response

# ============= CODE EXECUTION =============
def execute_code(code):
    """Execute Python code and capture output"""
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        namespace = {}
        exec(code, namespace)
        output = stdout_buffer.getvalue()
        errors = stderr_buffer.getvalue()
        return {"success": True, "output": output.strip() or None, "error": errors.strip() or None}
    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

# ============= MAIN DEMO CLASS =============
class CodeActDemo:
    def __init__(self, backend="auto", model_name=None, adapter_path=None):
        # Default model
        if model_name is None:
            model_name = "Qwen/Qwen2.5-3B"

        # Default adapter paths
        if adapter_path is None:
            adapter_path = "./models/codeact-mlx-qwen2.5-3b"

        # Auto-detect or use specified backend
        if backend == "auto":
            backend = detect_backend()

        print(f"\n{'='*60}")
        print(f"CodeAct Interactive Demo")
        print(f"Backend: {backend.upper()}")
        print(f"{'='*60}\n")

        self.backend_name = backend

        # Initialize backend
        if backend == "mlx":
            self.backend = MLXBackend(model_name, adapter_path)
        else:
            self.backend = PyTorchBackend(model_name, device=backend, adapter_path=adapter_path)

        self.tokenizer = self.backend.tokenizer if hasattr(self.backend, 'tokenizer') else None
        self.conversation_history = []

        self.system_prompt = """You are a Python coding assistant. Follow these rules STRICTLY:

1. UNDERSTAND THE PROBLEM FIRST:
   - Read the user's question carefully
   - Identify EXACTLY what output is expected
   - If asked for "first N items", return exactly N items
   - If asked for "prime numbers", return ONLY prime numbers (2, 3, 5, 7, 11, 13...)

2. USE THESE TAGS:
   - <thought>Step-by-step reasoning about the problem and expected output</thought>
   - <execute>Complete, working Python code that prints the answer</execute>
   - <solution>Clear statement of the final answer</solution>

3. VERIFY YOUR CODE:
   - Before writing code, think: "What should the output look like?"
   - For "first 10 primes": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
   - For "sum of 1 to 100": 5050
   - Make sure your code produces EXACTLY this output

4. WHEN USER SAYS OUTPUT IS WRONG:
   - Analyze WHY it was wrong
   - Identify the bug in your logic
   - Write CORRECTED code
   - Do NOT repeat the same mistake

5. COMMON MISTAKES TO AVOID:
   - Incomplete algorithms (e.g., sieve without marking composites)
   - Off-by-one errors
   - Returning wrong data type
   - Not printing the result"""

        print("Model loaded successfully!\n")

    def parse_response(self, response):
        """Extract tags from response"""
        parts = {'thought': None, 'execute': None, 'solution': None, 'feedback': None}
        for tag in parts:
            match = re.search(f'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
            if match:
                parts[tag] = match.group(1).strip()
        return parts

    def build_prompt(self, user_input, execution_result=None):
        """Build prompt with conversation history"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        if execution_result:
            content = f"Previous execution result: {execution_result}\n\nUser: {user_input}"
        else:
            content = user_input

        messages.append({"role": "user", "content": content})

        # Apply chat template
        if hasattr(self.backend, 'tokenizer') and hasattr(self.backend.tokenizer, 'apply_chat_template'):
            return self.backend.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

    def chat(self, user_input, execution_result=None):
        """Generate response"""
        prompt = self.build_prompt(user_input, execution_result)
        return self.backend.generate(prompt, max_tokens=400)

    def run(self):
        """Run interactive loop"""
        print("="*60)
        print(f"Running on: {self.backend_name.upper()}")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - 'clear' - Clear conversation history")
        print("  - 'quit' - Exit")
        print("="*60 + "\n")

        last_execution_result = None

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    last_execution_result = None
                    print("Conversation cleared")
                    continue

                print("\n[Generating...]", end=" ", flush=True)
                response = self.chat(user_input, last_execution_result)
                print("Done!\n")

                parts = self.parse_response(response)

                if parts['thought']:
                    print(f"Thought:\n{parts['thought']}\n")

                if parts['execute']:
                    print(f"Code:\n```python\n{parts['execute']}\n```\n")
                    print("Executing...\n")

                    result = execute_code(parts['execute'])

                    if result["success"]:
                        if result["output"]:
                            print(f"Output:\n{result['output']}")
                            last_execution_result = f"Output: {result['output']}"

                            print("\n" + "-"*40)
                            feedback = input("Is this correct? (y/n/skip): ").strip().lower()

                            if feedback == 'n':
                                # Ask for more details about what's wrong
                                print("\nWhat's wrong with the output?")
                                error_detail = input("(e.g., 'should only have primes', 'wrong count'): ").strip()
                                if error_detail:
                                    last_execution_result = f"OUTPUT WAS WRONG! Got: {result['output'][:200]}. Problem: {error_detail}. Please FIX the code."
                                else:
                                    last_execution_result = f"OUTPUT WAS WRONG! Got: {result['output'][:200]}. Please analyze the bug and FIX it."
                                print("\nWill try to fix on next query...")

                                # Add to history with error context
                                self.conversation_history.append({"role": "user", "content": user_input})
                                self.conversation_history.append({"role": "assistant", "content": response})
                                self.conversation_history.append({"role": "user", "content": f"ERROR: Your code output was incorrect. {error_detail if error_detail else 'Please fix it.'}"})
                            elif feedback == 'y':
                                print("\nCorrect!")
                                last_execution_result = None
                                self.conversation_history.append({"role": "user", "content": user_input})
                                self.conversation_history.append({"role": "assistant", "content": response})
                            else:
                                last_execution_result = None
                        else:
                            print("Code executed (no output)")
                            last_execution_result = None

                        if result["error"]:
                            print(f"Warnings: {result['error']}")
                    else:
                        print(f"Error: {result['error']}")
                        last_execution_result = f"Error: {result['error']}"

                if parts['solution']:
                    print(f"\nSolution:\n{parts['solution']}")

                if parts['feedback']:
                    print(f"\nFeedback:\n{parts['feedback']}")

                if not any(parts.values()):
                    print(f"Response:\n{response[:500]}")

                # Limit history
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]

                print("\n" + "="*60)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="CodeAct Interactive Demo")
    parser.add_argument("--backend", choices=["auto", "cuda", "mps", "mlx", "cpu"],
                       default="auto", help="Backend to use (default: auto)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B",
                       help="Model name or path")
    parser.add_argument("--adapter", type=str, default=None,
                       help="Path to LoRA adapter")

    args = parser.parse_args()

    demo = CodeActDemo(
        backend=args.backend,
        model_name=args.model,
        adapter_path=args.adapter
    )
    demo.run()

if __name__ == "__main__":
    main()
