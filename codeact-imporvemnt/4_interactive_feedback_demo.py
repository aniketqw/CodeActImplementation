"""
Step 4: Interactive Demo with Code Execution using MLX on Apple Silicon
"""
import re
import sys
import os
from io import StringIO

class CodeActDemo:
    def __init__(self, model_name="Qwen/Qwen2-0.5B", adapter_path="./models/codeact-mlx-qwen2-0.5b"):
        print("Loading MLX model...")

        from mlx_lm import load, generate

        self.generate_fn = generate

        # Check if adapter exists
        if os.path.exists(adapter_path):
            print(f"Loading with LoRA adapter: {adapter_path}")
            self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        else:
            print(f"Adapter not found at {adapter_path}, loading base model")
            self.model, self.tokenizer = load(model_name)

        self.model_name = model_name
        self.conversation_history = []

        self.system_prompt = """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""

        print("Model ready on Apple Silicon!\n")

    def execute_code(self, code):
        """Execute Python code using exec() and capture output"""
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer

            # Create namespace for code execution
            namespace = {}

            # Execute the code with exec()
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

    def parse_response(self, response):
        """Parse and extract tags from response"""
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

    def chat(self, user_input, execution_result=None):
        """Generate response for user input with conversation history"""

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        if execution_result:
            messages.append({
                "role": "user",
                "content": f"Previous execution result: {execution_result}\n\nUser feedback: {user_input}"
            })
        else:
            messages.append({
                "role": "user",
                "content": user_input
            })

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"

        # Generate with MLX
        response = self.generate_fn(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False
        )

        return response

    def run(self):
        """Run interactive demo with code execution"""
        print("="*60)
        print("CodeAct Interactive Demo - MLX on Apple Silicon")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - 'exec' - Execute the last code block")
        print("  - 'clear' - Clear conversation history")
        print("  - 'quit' - Exit")
        print("="*60 + "\n")

        last_code = None
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

                if user_input.lower() == 'exec' and last_code:
                    print("\nExecuting code with exec()...")
                    result = self.execute_code(last_code)
                    if result["success"]:
                        if result["output"]:
                            print(f"Output:\n{result['output']}")
                            last_execution_result = f"Code output: {result['output']}"
                        else:
                            print("Code executed successfully (no output)")
                        if result["error"]:
                            print(f"Warnings: {result['error']}")
                    else:
                        print(f"Execution Error: {result['error']}")
                        last_execution_result = f"Execution error: {result['error']}"
                    continue

                print("\n[Generating...]", end=" ", flush=True)
                response = self.chat(user_input, last_execution_result)
                print("Done!\n")

                # Parse response
                parts = self.parse_response(response)

                # Display formatted response
                if parts['thought']:
                    print(f"Thought:\n{parts['thought']}\n")

                if parts['execute']:
                    last_code = parts['execute']
                    print(f"Code:\n```python\n{parts['execute']}\n```\n")

                    # Auto-execute the code
                    print("Executing code with exec()...\n")
                    result = self.execute_code(last_code)

                    if result["success"]:
                        if result["output"]:
                            print(f"Output:\n{result['output']}")
                            last_execution_result = f"Code output: {result['output']}"

                            # Ask for feedback
                            print("\n" + "-"*40)
                            feedback = input("Is this correct? (y/n/skip): ").strip().lower()

                            if feedback == 'n':
                                print("\nMarked as incorrect - will try to fix on next query")
                                last_execution_result += " [USER MARKED AS INCORRECT]"
                                self.conversation_history.append({"role": "user", "content": user_input})
                                self.conversation_history.append({"role": "assistant", "content": response})
                            elif feedback == 'y':
                                print("\nGreat! Correct answer.")
                                last_execution_result = None
                                self.conversation_history.append({"role": "user", "content": user_input})
                                self.conversation_history.append({"role": "assistant", "content": response})
                            else:
                                last_execution_result = None
                        else:
                            print("Code executed successfully (no output)")
                            last_execution_result = None

                        if result["error"]:
                            print(f"\nWarnings: {result['error']}")
                    else:
                        print(f"Execution Error: {result['error']}")
                        last_execution_result = f"Execution error: {result['error']}"

                if parts['solution']:
                    print(f"\nSolution:\n{parts['solution']}")

                if parts['feedback']:
                    print(f"\nSelf-Feedback:\n{parts['feedback']}")

                # If no structured output, print raw
                if not any(parts.values()):
                    print(f"Response:\n{response[:500]}")

                # Keep history manageable
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
                print("Please try again.")

if __name__ == "__main__":
    demo = CodeActDemo()
    demo.run()
