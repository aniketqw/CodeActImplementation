# CodeAct Fine-tuning Demo

**Fine-tune Qwen2-0.5B to generate and execute Python code with reasoning.**

Interactive demo with conversation memory, execution feedback loop, and real-time code execution.

---

## 🚀 Quick Start

### Use Pre-trained Model (Instant)
```bash
./interactive.sh
```

### Train Your Own Model (~10 minutes)
```bash
source activate codeact
./run_all.sh
```

---

## 📁 Project Structure

```
codeact-demo/
├── 1_create_dataset.py          # Creates training dataset (8 examples)
├── 2_finetune_model.py           # Fine-tunes Qwen2-0.5B (3 epochs)
├── 3_test_model.py               # Tests model with code execution
├── 4_interactive_demo.py         # Interactive chat with memory & feedback
├── run_all.sh                    # Runs full training pipeline (1→2→3)
├── interactive.sh                # Launches interactive demo
├── README.md                     # This file
├── data/                         # Training dataset (auto-created)
│   └── codeact_train.jsonl      # 8 CodeAct examples
├── models/                       # Trained models
│   └── codeact-qwen2-0.5b-final/  # Fine-tuned model (1.8GB)
└── codeact/                      # Conda environment
```

---

## ✨ Features

### 1. Code Generation with Reasoning
Model generates responses in CodeAct format:
```xml
<thought>reasoning process</thought>
<execute>
python code here
</execute>
```

### 2. Real Code Execution
- Extracts Python code from `<execute>` tags
- Executes using `exec()`
- Captures output and errors
- Shows actual results

### 3. Conversation Memory
- Remembers all questions and answers
- Tracks code generated and execution results
- Maintains full conversation context

### 4. Execution Feedback Loop
When you mark output as wrong (`n`):
- System saves execution result
- Next prompt includes: `"Previous output: X [USER MARKED AS INCORRECT]"`
- Model can learn from mistakes and try again

---

## 🎮 Interactive Demo Commands

| Command | Action |
|---------|--------|
| Type question | Model generates code and executes it |
| `y` | Mark output as correct, save to history |
| `n` | Mark output as wrong, include feedback in next prompt |
| `skip` | Don't save to history |
| `reset` | Clear conversation history |
| `clear` | Clear screen |
| `quit` / `exit` | Exit demo |

---

## 📖 Usage Examples

### Example 1: Correct Answer
```
You: Calculate 5+5

💭 Thought: I'll add 5 and 5.

💻 Generated Code:
------------------------------------------------------------
result = 5 + 5
print(f"Result: {result}")
------------------------------------------------------------

⚙️  Executing code...

✅ Output:
Result: 10

Is this correct? (y/n/skip): y

✨ Great! The model solved it correctly!
```

### Example 2: Wrong Answer with Feedback
```
You: give me first 10 prime numbers

💭 Thought: I need to find prime numbers.

💻 Generated Code:
------------------------------------------------------------
primes = [n for n in range(10) if is_prime(n)]
------------------------------------------------------------

✅ Output:
[2, 3, 5, 7]

Is this correct? (y/n/skip): n

💡 The code executed but produced WRONG results.
   The next response will try to fix it based on this execution!

============================================================

You: I want FIRST 10 primes, not primes till 10

🤖 Model now receives:
   - Previous execution: [2, 3, 5, 7] [USER MARKED AS INCORRECT]
   - Your clarification
   - Full conversation history

   It can generate better code based on this feedback!
```

---

## 🔧 Training Pipeline

### Full Pipeline (Automatic)
```bash
source activate codeact
./run_all.sh
```

This runs:
1. Creates dataset (8 examples)
2. Fine-tunes model (3 epochs, ~10 minutes)
3. Tests model (5 test cases with execution)

### Individual Steps

**Step 1: Create Dataset**
```bash
python 1_create_dataset.py
```
- Creates 8 CodeAct training examples
- Saves to `data/codeact_train.jsonl`
- Examples include: sum, primes, average, reverse, palindrome, factorial, sort, max

**Step 2: Fine-tune Model**
```bash
python 2_finetune_model.py
```
- Downloads Qwen2-0.5B from HuggingFace
- Fine-tunes for 3 epochs
- Saves to `models/codeact-qwen2-0.5b-final/`
- Takes ~10 minutes on CPU

**Step 3: Test Model**
```bash
python 3_test_model.py
```
- Runs 5 test cases
- Executes generated code
- Shows thought process, code, and results

---

## 💡 How It Works

### CodeAct Format
The model is trained to generate responses with two components:

1. **`<thought>`** - Reasoning process
   ```xml
   <thought>I need to calculate the sum of 1 to 10.</thought>
   ```

2. **`<execute>`** - Python code to run
   ```xml
   <execute>
   numbers = range(1, 11)
   result = sum(numbers)
   print(f"Sum: {result}")
   </execute>
   ```

### Execution Process
1. Model generates `<thought>` and `<execute>` tags
2. System extracts code using regex
3. Code runs in isolated namespace with `exec()`
4. Output/errors captured and displayed
5. User provides feedback (correct/wrong)
6. Feedback incorporated in next prompt

### Feedback Loop
```
Question → Generate Code → Execute → Show Results → Get Feedback
                ↑                                        ↓
                └────────── Include in next prompt ──────┘
```

---

## 🛠️ Technical Details

### Model
- **Base:** Qwen/Qwen2-0.5B (494M parameters)
- **Training:** 8 examples, 3 epochs
- **Loss:** 2.30 → 0.08 (good convergence)
- **Format:** CodeAct pattern (`<thought>` + `<execute>`)

### Environment
- **Python:** 3.10+
- **Device:** CPU (MPS/CUDA if available)
- **Inference:** 5-30 seconds per response on CPU
- **Memory:** ~2GB RAM

### Dependencies
```
torch
transformers
datasets
accelerate
numpy<2
```

All pre-installed in `codeact` conda environment.

---

## 🎯 Training Dataset

The dataset contains 8 examples covering:
- Sum of numbers
- Prime number finding
- Average calculation
- List reversal
- Palindrome checking
- Factorial calculation
- List sorting
- Finding maximum value

Each example shows:
- System prompt (defines CodeAct format)
- User question
- Assistant response with `<thought>` and `<execute>` tags

---

## 📊 Performance

**Training Results:**
- Initial loss: 2.30
- Final loss: 0.08
- Training time: ~7 minutes on Apple M1/M2 CPU
- Model size: 1.8GB

**Test Results:**
All 5 test cases successfully:
- Generate proper `<thought>` and `<execute>` tags
- Execute code without errors
- Produce correct outputs

---

## 🚨 Important Notes

### Warnings Suppressed
The code suppresses these warnings:
- `torch_dtype` deprecation (uses `dtype` instead)
- Tokenizer regex pattern (known HuggingFace issue)

### Code Execution Safety
- Code runs in isolated namespace
- No access to system files by default
- Captures stdout/stderr safely
- Handles execution errors gracefully

### Model Limitations
- Small model (0.5B) - may generate incorrect logic
- Limited training data (8 examples)
- CPU inference is slow (5-30 sec)
- May need multiple attempts for complex tasks

### Improvements from Feedback
The feedback loop helps the model:
- See what output was wrong
- Understand user clarifications
- Try alternative approaches
- Learn from execution results

---

## 🔄 Retraining

To retrain the model:

1. **Modify dataset:**
   Edit `1_create_dataset.py` to add more examples

2. **Retrain:**
   ```bash
   source activate codeact
   ./run_all.sh
   ```

3. **Test new model:**
   ```bash
   ./interactive.sh
   ```

The model will overwrite `models/codeact-qwen2-0.5b-final/`.

---

## 📝 Tips & Tricks

1. **Model gives wrong answer?**
   - Mark as `n` and clarify what you want
   - Next attempt includes your feedback

2. **Want to start fresh?**
   - Type `reset` to clear conversation history

3. **Code execution fails?**
   - Check if code needs imports (model may forget)
   - Simplify your question

4. **Improve results?**
   - Add more training examples in step 1
   - Retrain with `./run_all.sh`

5. **Speed up inference?**
   - Use GPU if available (automatically detected)
   - Reduce `max_new_tokens` in demo script

---

## 🎓 Learning Resources

**CodeAct Pattern:**
- Combines reasoning (`<thought>`) with action (`<execute>`)
- Allows verification through execution
- Catches logical errors through feedback

**Fine-tuning:**
- Small dataset (8 examples) is enough for simple tasks
- More epochs may overfit on small datasets
- Loss should decrease steadily

**Execution Feedback:**
- Helps model learn from mistakes
- Provides ground truth through actual execution
- Creates interactive learning loop

---

## 🤝 Contributing Ideas

Want to improve the model? Try:
1. Add more diverse training examples
2. Increase training epochs (3→5)
3. Use larger base model (Qwen2-1.5B)
4. Add validation dataset
5. Implement temperature tuning
6. Add code analysis before execution

---

## 📄 License

This is a demo project for educational purposes.
- Base model (Qwen2-0.5B) follows Qwen license
- Demo code can be used freely

---

## 🙏 Acknowledgments

- **Qwen Team** - Base model
- **HuggingFace** - Transformers library
- **CodeAct** - Inspiration for the pattern

---

**Made with ❤️ for learning AI code generation**

Enjoy experimenting with your CodeAct model! 🚀
