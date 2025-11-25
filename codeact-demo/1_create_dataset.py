"""
Step 1: Create CodeAct Training Dataset
"""
import json
import os

def create_codeact_dataset():
    """Create a dataset of CodeAct examples"""

    print("\n" + "="*60)
    print("Creating CodeAct Training Dataset")
    print("="*60)

    # Training examples with CodeAct format
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Calculate the sum of first 10 natural numbers."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I need to sum the numbers 1 to 10.</thought>\n\n<execute>\nnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nresult = sum(numbers)\nprint(f\"Sum: {result}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Find all prime numbers between 1 and 20."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I need to find all prime numbers between 1 and 20.</thought>\n\n<execute>\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nresult = [n for n in range(1, 21) if is_prime(n)]\nprint(f\"Prime numbers: {result}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Calculate the average of [15, 25, 35, 45, 55]."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I'll sum all numbers and divide by count.</thought>\n\n<execute>\nnumbers = [15, 25, 35, 45, 55]\naverage = sum(numbers) / len(numbers)\nprint(f\"Average: {average}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Reverse the list [1, 2, 3, 4, 5]."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I'll reverse the list using slicing.</thought>\n\n<execute>\nnumbers = [1, 2, 3, 4, 5]\nreversed_numbers = numbers[::-1]\nprint(f\"Reversed: {reversed_numbers}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Check if 'racecar' is a palindrome."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I'll check if the string equals its reverse.</thought>\n\n<execute>\ndef is_palindrome(s):\n    return s == s[::-1]\n\nresult = is_palindrome('racecar')\nprint(f\"Is 'racecar' a palindrome? {result}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Calculate factorial of 5."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I'll calculate 5! = 5*4*3*2*1.</thought>\n\n<execute>\nimport math\nresult = math.factorial(5)\nprint(f\"Factorial of 5: {result}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Sort the list [5, 2, 8, 1, 9] in ascending order."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I'll use the sorted() function to sort the list.</thought>\n\n<execute>\nnumbers = [5, 2, 8, 1, 9]\nsorted_numbers = sorted(numbers)\nprint(f\"Sorted: {sorted_numbers}\")\n</execute>"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can execute Python code. Use <execute>code</execute> for code and <thought>reasoning</thought> for thinking."
                },
                {
                    "role": "user",
                    "content": "Find the maximum value in [10, 25, 5, 30, 15]."
                },
                {
                    "role": "assistant",
                    "content": "<thought>I'll use the max() function to find the largest value.</thought>\n\n<execute>\nnumbers = [10, 25, 5, 30, 15]\nmax_value = max(numbers)\nprint(f\"Maximum: {max_value}\")\n</execute>"
                }
            ]
        }
    ]

    # Save to JSONL file
    output_file = "data/codeact_train.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Created dataset with {len(examples)} examples")
    print(f"✓ Saved to: {output_file}")

    # Show file size
    file_size = os.path.getsize(output_file)
    print(f"✓ File size: {file_size / 1024:.2f} KB")

if __name__ == "__main__":
    create_codeact_dataset()
