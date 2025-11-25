"""
Test the feedback-aware model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FeedbackCodeActModel:
    def __init__(self, model_path="./models/codeact-feedback-qwen2-0.5b-final"):
        print(f"Loading model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.system_prompt = """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
        
        print("✓ Model loaded!")
    
    def generate(self, prompt, max_tokens=400):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
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
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response
    
    def simulate_execution(self, code):
        """Simulate code execution for testing"""
        try:
            # Create a safe execution environment
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            return "Success"
        except Exception as e:
            return f"Error: {str(e)}"

def run_tests():
    """Run comprehensive tests"""
    
    print("\n" + "="*60)
    print("Testing Feedback-Aware CodeAct Model")
    print("="*60 + "\n")
    
    model = FeedbackCodeActModel()
    
    test_cases = [
        "Calculate the sum of first 5 fibonacci numbers.",
        "Find all divisors of 36.",
        "Check if 'A man a plan a canal Panama' is a palindrome (ignore case and spaces).",
        "Calculate the mean and standard deviation of [2, 4, 6, 8, 10].",
        "Find the longest word in 'The quick brown fox jumps over the lazy dog'."
    ]
    
    results = []
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}")
        print("="*60)
        print(f"\nPrompt: {prompt}")
        print("\nResponse:")
        print("-"*60)
        
        response = model.generate(prompt)
        print(response)
        
        # Extract and analyze feedback if present
        if '<feedback>' in response and '</feedback>' in response:
            feedback_start = response.find('<feedback>') + len('<feedback>')
            feedback_end = response.find('</feedback>')
            feedback = response[feedback_start:feedback_end].strip()
            
            print("\n" + "-"*60)
            print("Extracted Self-Feedback:")
            print("-"*60)
            print(feedback)
            
            # Try to extract score
            for line in feedback.split('\n'):
                if 'score:' in line.lower():
                    try:
                        score = int(line.split(':')[1].strip())
                        results.append(score)
                    except:
                        pass
        
        print("-"*60)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total tests: {len(test_cases)}")
    if results:
        print(f"Self-reported scores: {results}")
        print(f"Average self-score: {sum(results)/len(results):.1f}")
    print("="*60)

if __name__ == "__main__":
    run_tests()