"""
Step 3: Test the Fine-tuned Model with Code Execution
"""
import torch
import re
import sys
from io import StringIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path="./models/codeact-qwen2-0.5b-final"):
    """Load the fine-tuned model"""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    print(f"✓ Model loaded successfully")
    return model, tokenizer

def execute_code(code):
    """Execute Python code and capture output"""
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        namespace = {}
        exec(code, namespace)

        output = stdout_buffer.getvalue()
        errors = stderr_buffer.getvalue()

        return {
            "success": True,
            "output": output.strip() if output else None,
            "error": errors.strip() if errors else None
        }

    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": str(e)
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def extract_tags(response):
    """Extract <thought> and <execute> tags from response"""
    thought = None
    code = None

    thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    code_match = re.search(r'<execute>(.*?)</execute>', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    return thought, code

def generate_response(model, tokenizer, prompt, max_tokens=300):
    """Generate a response from the model"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    return response

def test_model():
    """Run test cases on the model"""

    print("\n" + "="*60)
    print("Loading Fine-tuned Model")
    print("="*60 + "\n")

    model, tokenizer = load_model()

    # Test cases
    test_cases = [
        "Calculate the sum of first 10 natural numbers.",
        "Find all prime numbers between 1 and 20.",
        "Calculate the average of [15, 25, 35, 45, 55].",
        "Reverse the list [1, 2, 3, 4, 5].",
        "Check if 'racecar' is a palindrome.",
    ]

    print("="*60)
    print("Running Test Cases with Code Execution")
    print("="*60)

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}\n")

        # Generate response
        response = generate_response(model, tokenizer, prompt)

        # Extract thought and code
        thought, code = extract_tags(response)

        # Display thought
        if thought:
            print(f"💭 Thought: {thought}\n")

        # Display and execute code
        if code:
            print("💻 Generated Code:")
            print("-"*60)
            print(code)
            print("-"*60)
            print("\n⚙️  Executing...\n")

            # Execute
            result = execute_code(code)

            if result["success"]:
                if result["output"]:
                    print("✅ Output:")
                    print(result["output"])
                else:
                    print("✅ Code executed successfully (no output)")

                if result["error"]:
                    print(f"\n⚠️  Warnings:\n{result['error']}")
            else:
                print(f"❌ Execution Error:\n{result['error']}")
        else:
            print("📝 Raw Response:")
            print(response[:300])

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)

if __name__ == "__main__":
    test_model()
