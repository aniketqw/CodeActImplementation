"""
Step 4: Interactive Demo with Code Execution
"""
import torch
import re
import sys
from io import StringIO
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeActDemo:
    def __init__(self, model_path="./models/codeact-qwen2-0.5b-final"):
        print("Loading model...")

        import warnings
        warnings.filterwarnings('ignore', message='.*torch_dtype.*')
        warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Conversation history
        self.conversation_history = []

        print("✓ Model ready!\n")

    def execute_code(self, code):
        """Execute Python code and capture output"""
        # Create a string buffer to capture output
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        # Save original stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # Redirect stdout/stderr to capture output
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer

            # Create a namespace for code execution
            namespace = {}

            # Execute the code
            exec(code, namespace)

            # Get the output
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
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def extract_tags(self, response):
        """Extract <thought> and <execute> tags from response"""
        thought = None
        code = None

        # Extract thought
        thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract code
        code_match = re.search(r'<execute>(.*?)</execute>', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        return thought, code

    def chat(self, user_input, execution_result=None):
        """Generate response for user input with conversation history"""

        # Build messages with history
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking. Learn from execution results and user feedback to improve your answers."
            }
        ]

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add execution feedback if available
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
                max_new_tokens=300,
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

    def run(self):
        """Run interactive demo"""
        print("="*60)
        print("CodeAct Interactive Demo - WITH MEMORY & FEEDBACK")
        print("="*60)
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • Type 'quit' or 'exit' to exit")
        print("  • Type 'clear' to clear screen")
        print("  • Type 'reset' to clear conversation history")
        print("="*60 + "\n")

        last_execution_result = None

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break

                if user_input.lower() == 'clear':
                    import os
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue

                if user_input.lower() == 'reset':
                    self.conversation_history = []
                    last_execution_result = None
                    print("\n✨ Conversation history cleared!\n")
                    continue

                # Get model response (with history and feedback)
                print("\n🤖 Thinking...", end=" ", flush=True)
                response = self.chat(user_input, last_execution_result)
                print("Done!\n")

                # Extract thought and code
                thought, code = self.extract_tags(response)

                # Display thought
                if thought:
                    print(f"💭 Thought: {thought}\n")

                # Display and execute code
                if code:
                    print("💻 Generated Code:")
                    print("-" * 60)
                    print(code)
                    print("-" * 60)
                    print("\n⚙️  Executing code...\n")

                    # Execute the code
                    result = self.execute_code(code)

                    if result["success"]:
                        if result["output"]:
                            print("✅ Output:")
                            print(result["output"])

                            # Store execution result for feedback
                            last_execution_result = f"Code output: {result['output']}"

                            # Ask if output is correct
                            print("\n" + "="*60)
                            feedback = input("Is this correct? (y/n/skip): ").strip().lower()

                            if feedback == 'n':
                                print("\n💡 The code executed but produced WRONG results.")
                                print("   The next response will try to fix it based on this execution!")
                                last_execution_result += " [USER MARKED AS INCORRECT]"

                                # Add to conversation history
                                self.conversation_history.append({
                                    "role": "user",
                                    "content": user_input
                                })
                                self.conversation_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                            elif feedback == 'y':
                                print("\n✨ Great! The model solved it correctly!")
                                last_execution_result = None

                                # Add to conversation history
                                self.conversation_history.append({
                                    "role": "user",
                                    "content": user_input
                                })
                                self.conversation_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                            else:
                                # Skip - don't add to history
                                last_execution_result = None
                        else:
                            print("✅ Code executed successfully (no output)")
                            last_execution_result = None

                        if result["error"]:
                            print(f"\n⚠️  Warnings:\n{result['error']}")
                    else:
                        print(f"❌ Execution Error:\n{result['error']}")
                        print("\n💡 The code has a syntax or runtime error.")
                        last_execution_result = f"Execution error: {result['error']}"
                else:
                    # No code to execute, just show the response
                    print("📝 Response:")
                    print(response[:500])  # Limit output

                print("\n" + "="*60 + "\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")

if __name__ == "__main__":
    demo = CodeActDemo()
    demo.run()
