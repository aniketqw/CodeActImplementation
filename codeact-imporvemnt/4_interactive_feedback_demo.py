"""
Interactive Demo with Feedback
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class InteractiveFeedbackDemo:
    def __init__(self, model_path="./models/codeact-feedback-qwen2-0.5b-final"):
        print("Loading model...")
        
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
        
        self.conversation_history = []
        print("✓ Model ready!\n")
    
    def chat(self, user_input):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
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
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def execute_code(self, code):
        """Safely execute extracted code"""
        try:
            import io
            import sys
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            exec(code)
            
            output = buffer.getvalue()
            sys.stdout = old_stdout
            
            return output if output else "Code executed successfully (no output)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def parse_response(self, response):
        """Parse and format the response"""
        parts = {
            'thought': None,
            'execute': None,
            'solution': None,
            'feedback': None
        }
        
        for tag in parts.keys():
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                parts[tag] = match.group(1).strip()
        
        return parts
    
    def run(self):
        print("="*60)
        print("CodeAct with Feedback - Interactive Demo")
        print("="*60)
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • 'exec' - Execute the last code block")
        print("  • 'clear' - Clear conversation history")
        print("  • 'quit' - Exit")
        print("="*60 + "\n")
        
        last_code = None
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Conversation cleared")
                    continue
                
                if user_input.lower() == 'exec' and last_code:
                    print("\n📟 Executing code...")
                    result = self.execute_code(last_code)
                    print(f"Result: {result}")
                    continue
                
                print("\n🤖 Assistant:")
                response = self.chat(user_input)
                
                # Parse response
                parts = self.parse_response(response)
                
                # Display formatted response
                if parts['thought']:
                    print(f"\n💭 Thought:\n{parts['thought']}")
                
                if parts['execute']:
                    last_code = parts['execute']
                    print(f"\n💻 Code:\n```python\n{parts['execute']}\n```")
                
                if parts['solution']:
                    print(f"\n✅ Solution:\n{parts['solution']}")
                
                if parts['feedback']:
                    print(f"\n📊 Self-Feedback:\n{parts['feedback']}")
                
                # If no structured output, print raw
                if not any(parts.values()):
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

if __name__ == "__main__":
    demo = InteractiveFeedbackDemo()
    demo.run()