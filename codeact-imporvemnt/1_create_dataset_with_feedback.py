"""
Enhanced CodeAct Dataset with Reinforcement Feedback
"""
import json
import os

def create_enhanced_dataset():
    """
    Create dataset with feedback signals for reinforcement learning.
    
    Feedback format:
    <feedback>
    score: 1-10
    correctness: correct/incorrect/partial
    efficiency: excellent/good/acceptable/poor
    explanation: why this score
    </feedback>
    """
    
    training_data = [
        # Example 1: Excellent response (score 10)
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Calculate the sum of squares from 1 to 5."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I need to calculate 1² + 2² + 3² + 4² + 5².
I'll use a list comprehension with sum() for efficiency.
</thought>

<execute>
result = sum(i**2 for i in range(1, 6))
print(f"Sum of squares: {result}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Sum of squares: 55"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
The sum of squares from 1 to 5 is 55.
</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Used efficient generator expression, correct result, clear code.
</feedback>"""
                }
            ]
        },
        
        # Example 2: Good response with minor improvement possible (score 8)
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Find all prime numbers between 1 and 20."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I'll check each number from 2 to 20 for primality.
A number is prime if it's only divisible by 1 and itself.
</thought>

<execute>
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(1, 21) if is_prime(n)]
print(f"Primes: {primes}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Primes: [2, 3, 5, 7, 11, 13, 17, 19]"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Prime numbers between 1 and 20: [2, 3, 5, 7, 11, 13, 17, 19]
</solution>

<feedback>
score: 8
correctness: correct
efficiency: acceptable
explanation: Correct result but could optimize by only checking up to sqrt(n). Good code structure.
</feedback>"""
                }
            ]
        },
        
        # Example 3: Response with error, then self-correction (shows learning)
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Calculate factorial of 5 using recursion."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I'll write a recursive factorial function.
Base case: 0! = 1
Recursive case: n! = n * (n-1)!
</thought>

<execute>
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n)  # Bug: should be n-1
    
print(f"5! = {factorial(5)}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: RecursionError: maximum recursion depth exceeded"
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I made an error - the recursive call should use (n-1) not (n).
Let me fix this.
</thought>

<execute>
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)  # Fixed: now uses n-1
    
print(f"5! = {factorial(5)}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: 5! = 120"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
5! = 120
</solution>

<feedback>
score: 7
correctness: correct
efficiency: good
explanation: Initial error but self-corrected successfully. Shows debugging ability. Final answer correct.
</feedback>"""
                }
            ]
        },
        
        # Example 4: Excellent efficient solution (score 10)
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Find the maximum and minimum in [3, 7, 1, 9, 4, 2] in one pass."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
For efficiency, I'll iterate once and track both max and min.
This is O(n) compared to calling max() and min() separately which is O(2n).
</thought>

<execute>
numbers = [3, 7, 1, 9, 4, 2]
min_val = max_val = numbers[0]

for num in numbers[1:]:
    if num < min_val:
        min_val = num
    elif num > max_val:
        max_val = num

print(f"Min: {min_val}, Max: {max_val}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Min: 1, Max: 9"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Minimum: 1, Maximum: 9
</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Single-pass O(n) solution as requested. Clear logic, efficient implementation.
</feedback>"""
                }
            ]
        },
        
        # Example 5: Average solution with room for improvement (score 6)
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Sort a list [5, 2, 8, 1, 9] in ascending order."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I'll implement bubble sort to sort the list.
</thought>

<execute>
lst = [5, 2, 8, 1, 9]
n = len(lst)
for i in range(n):
    for j in range(0, n-i-1):
        if lst[j] > lst[j+1]:
            lst[j], lst[j+1] = lst[j+1], lst[j]
print(f"Sorted: {lst}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Sorted: [1, 2, 5, 8, 9]"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Sorted list: [1, 2, 5, 8, 9]
</solution>

<feedback>
score: 6
correctness: correct
efficiency: poor
explanation: Correct result but bubble sort is O(n²). Should use built-in sorted() which is O(n log n) for practical use.
</feedback>"""
                }
            ]
        },
        
        # Example 6: String manipulation with excellent solution
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Check if 'racecar' is a palindrome."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
A palindrome reads the same forwards and backwards.
I'll compare the string with its reverse using slicing.
</thought>

<execute>
word = 'racecar'
is_palindrome = word == word[::-1]
print(f"'{word}' is palindrome: {is_palindrome}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: 'racecar' is palindrome: True"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Yes, 'racecar' is a palindrome.
</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Pythonic solution using slicing. Clean, readable, and efficient O(n).
</feedback>"""
                }
            ]
        },
        
        # Example 7: Math problem with good solution
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Calculate the GCD of 48 and 18."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I'll use the Euclidean algorithm for GCD.
It's efficient and well-established.
</thought>

<execute>
import math
gcd = math.gcd(48, 18)
print(f"GCD(48, 18) = {gcd}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: GCD(48, 18) = 6"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
GCD(48, 18) = 6
</solution>

<feedback>
score: 9
correctness: correct
efficiency: excellent
explanation: Used standard library for optimal implementation. Could have shown manual implementation for learning purposes.
</feedback>"""
                }
            ]
        },
        
        # Example 8: List comprehension mastery
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Get squares of even numbers from 1 to 10."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I'll use a list comprehension with a filter for even numbers.
This is concise and Pythonic.
</thought>

<execute>
squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Squares of even numbers: {squares}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Squares of even numbers: [4, 16, 36, 64, 100]"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Squares of even numbers (1-10): [4, 16, 36, 64, 100]
</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Perfect use of list comprehension with filtering. Pythonic, readable, efficient.
</feedback>"""
                }
            ]
        },
        
        # Example 9: Error handling demonstration
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Safely divide 10 by 0."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
Division by zero raises an error.
I'll use try-except to handle it gracefully.
</thought>

<execute>
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"

result = safe_divide(10, 0)
print(f"Result: {result}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Result: Cannot divide by zero"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Division by zero handled safely - returns error message instead of crashing.
</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Proper error handling with try-except. Production-quality code pattern.
</feedback>"""
                }
            ]
        },
        
        # Example 10: Dictionary manipulation
        {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that executes Python code.
Use these tags:
- <thought>reasoning</thought> for thinking
- <execute>code</execute> for code
- <solution>answer</solution> for final answer
- <feedback>assessment</feedback> for self-evaluation"""
                },
                {
                    "role": "user",
                    "content": "Count character frequency in 'hello world'."
                },
                {
                    "role": "assistant",
                    "content": """<thought>
I'll use collections.Counter for efficient counting.
It's the standard way to count frequencies in Python.
</thought>

<execute>
from collections import Counter
text = 'hello world'
freq = Counter(text)
print(f"Frequency: {dict(freq)}")
</execute>"""
                },
                {
                    "role": "user",
                    "content": "Execution result: Frequency: {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}"
                },
                {
                    "role": "assistant",
                    "content": """<solution>
Character frequencies: h=1, e=1, l=3, o=2, space=1, w=1, r=1, d=1
</solution>

<feedback>
score: 10
correctness: correct
efficiency: excellent
explanation: Used Counter from collections - the idiomatic Python solution. Clean and efficient.
</feedback>"""
                }
            ]
        }
    ]
    
    # Save dataset
    os.makedirs('data', exist_ok=True)
    output_file = 'data/codeact_feedback_train.jsonl'
    
    with open(output_file, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print("="*60)
    print(f"✓ Total examples: {len(training_data)}")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Analyze feedback scores
    scores = []
    for item in training_data:
        for msg in item['messages']:
            if msg['role'] == 'assistant' and 'score:' in msg['content']:
                # Extract score
                for line in msg['content'].split('\n'):
                    if 'score:' in line:
                        try:
                            score = int(line.split(':')[1].strip())
                            scores.append(score)
                        except:
                            pass
    
    if scores:
        print(f"\nFeedback Score Distribution:")
        print(f"  Average: {sum(scores)/len(scores):.1f}")
        print(f"  Min: {min(scores)}")
        print(f"  Max: {max(scores)}")
        print(f"  Scores: {sorted(scores)}")
    
    return output_file

if __name__ == "__main__":
    print("="*60)
    print("Creating Enhanced CodeAct Dataset with Feedback")
    print("="*60)
    create_enhanced_dataset()