"""
Create Expanded CodeAct Training Dataset with 100 Examples
Categories: Math, Strings, Lists, Algorithms, Data Structures (20 each)
"""
import json
import os

SYSTEM_PROMPT = """You are a Python coding assistant. Follow these rules STRICTLY:

1. UNDERSTAND THE PROBLEM FIRST:
   - Read the user's question carefully
   - Identify EXACTLY what output is expected
   - If asked for "first N items", return exactly N items
   - If asked for "prime numbers", return ONLY prime numbers

2. USE THESE TAGS:
   - <thought>Step-by-step reasoning</thought>
   - <execute>Complete Python code</execute>
   - <solution>Final answer</solution>

3. VERIFY YOUR CODE produces the correct output before submitting.

4. WHEN USER SAYS OUTPUT IS WRONG:
   - Analyze WHY it was wrong
   - Write CORRECTED code
   - Do NOT repeat the same mistake"""

def create_example(task, thought, code, output, solution, score, correctness, efficiency, explanation):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"<thought>{thought}</thought>\n\n<execute>\n{code}\n</execute>\n\n<solution>{solution}</solution>"},
            {"role": "user", "content": f"Execution result: {output}"},
            {"role": "assistant", "content": f"<feedback>\nscore: {score}\ncorrectness: {correctness}\nefficiency: {efficiency}\nexplanation: {explanation}\n</feedback>"}
        ]
    }

# ============= MATH EXAMPLES (20) =============
MATH_EXAMPLES = [
    ("Calculate the sum of numbers from 1 to 100", "Use formula n*(n+1)/2", "n = 100\nprint(n * (n + 1) // 2)", "5050", "Sum is 5050", 10, "correct", "excellent", "O(1) formula"),
    ("Calculate factorial of 10", "Use math.factorial", "import math\nprint(math.factorial(10))", "3628800", "10! = 3628800", 10, "correct", "excellent", "Built-in function"),
    ("Find the 15th Fibonacci number", "Iterative approach", "a, b = 0, 1\nfor _ in range(15): a, b = b, a + b\nprint(a)", "610", "F(15) = 610", 10, "correct", "excellent", "O(n) iterative"),
    ("Find GCD of 48 and 18", "Use math.gcd", "import math\nprint(math.gcd(48, 18))", "6", "GCD is 6", 10, "correct", "excellent", "Built-in gcd"),
    ("Find LCM of 12 and 18", "LCM = a*b/gcd(a,b)", "import math\na, b = 12, 18\nprint((a * b) // math.gcd(a, b))", "36", "LCM is 36", 10, "correct", "excellent", "Formula with gcd"),
    ("Check if 97 is prime", "Check divisibility to sqrt", "n = 97\nprint(all(n % i != 0 for i in range(2, int(n**0.5) + 1)))", "True", "97 is prime", 10, "correct", "excellent", "O(sqrt(n)) check"),
    ("Calculate 2^10", "Use exponentiation", "print(2 ** 10)", "1024", "2^10 = 1024", 10, "correct", "excellent", "Simple power"),
    ("Find average of [10, 20, 30, 40, 50]", "Sum divided by count", "nums = [10, 20, 30, 40, 50]\nprint(sum(nums) / len(nums))", "30.0", "Average is 30", 10, "correct", "excellent", "Built-in functions"),
    ("Calculate (7^100) mod 13", "Use pow with 3 args", "print(pow(7, 100, 13))", "9", "Result is 9", 10, "correct", "excellent", "Modular exponentiation"),
    ("Find square root of 144", "Use math.sqrt", "import math\nprint(int(math.sqrt(144)))", "12", "sqrt(144) = 12", 10, "correct", "excellent", "Math library"),
    ("Sum of squares 1 to 10", "Use formula n(n+1)(2n+1)/6", "n = 10\nprint(n * (n + 1) * (2 * n + 1) // 6)", "385", "Sum is 385", 10, "correct", "excellent", "O(1) formula"),
    ("Product of [1,2,3,4,5]", "Use reduce", "from functools import reduce\nprint(reduce(lambda x,y: x*y, [1,2,3,4,5]))", "120", "Product is 120", 10, "correct", "excellent", "Functional approach"),
    ("Absolute difference of -15 and 25", "Use abs()", "print(abs(-15 - 25))", "40", "Difference is 40", 10, "correct", "excellent", "Simple abs"),
    ("Check if 42 is even", "Use modulo", "print('even' if 42 % 2 == 0 else 'odd')", "even", "42 is even", 10, "correct", "excellent", "Modulo check"),
    ("Sum of digits of 12345", "Convert to string", "print(sum(int(d) for d in str(12345)))", "15", "Digit sum is 15", 10, "correct", "excellent", "String conversion"),
    ("Find all divisors of 24", "Check 1 to n", "n = 24\nprint([i for i in range(1, n+1) if n % i == 0])", "[1, 2, 3, 4, 6, 8, 12, 24]", "Divisors found", 10, "correct", "good", "O(n) approach"),
    ("Check if 49 is perfect square", "Use isqrt", "import math\nn = 49\nprint(math.isqrt(n) ** 2 == n)", "True", "49 is perfect square", 10, "correct", "excellent", "Integer sqrt"),
    ("Calculate 15% of 200", "Multiply and divide", "print(200 * 15 / 100)", "30.0", "15% of 200 is 30", 10, "correct", "excellent", "Simple percentage"),
    ("Round 3.14159 to 2 decimals", "Use round()", "print(round(3.14159, 2))", "3.14", "Rounded to 3.14", 10, "correct", "excellent", "Built-in round"),
    ("Ceiling and floor of 4.7", "Use math functions", "import math\nprint(f'ceil: {math.ceil(4.7)}, floor: {math.floor(4.7)}')", "ceil: 5, floor: 4", "Ceil=5, Floor=4", 10, "correct", "excellent", "Math functions"),
]

# ============= STRING EXAMPLES (20) =============
STRING_EXAMPLES = [
    ("Count 'a' in 'banana'", "Use count method", "print('banana'.count('a'))", "3", "3 occurrences", 10, "correct", "excellent", "Built-in count"),
    ("Reverse 'hello world'", "Use slicing", "print('hello world'[::-1])", "dlrow olleh", "Reversed string", 10, "correct", "excellent", "Slice reversal"),
    ("Check if 'racecar' is palindrome", "Compare with reverse", "s = 'racecar'\nprint(s == s[::-1])", "True", "Is palindrome", 10, "correct", "excellent", "Simple comparison"),
    ("Check if 'listen' and 'silent' are anagrams", "Sort and compare", "print(sorted('listen') == sorted('silent'))", "True", "Are anagrams", 10, "correct", "excellent", "Sorting approach"),
    ("Word frequency in 'the cat sat on the mat'", "Use Counter", "from collections import Counter\nprint(dict(Counter('the cat sat on the mat'.split())))", "{'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1}", "Frequencies counted", 10, "correct", "excellent", "Counter usage"),
    ("Find index of 'world' in 'hello world'", "Use find()", "print('hello world'.find('world'))", "6", "Index is 6", 10, "correct", "excellent", "Built-in find"),
    ("Convert 'Hello' to uppercase", "Use upper()", "print('Hello'.upper())", "HELLO", "Uppercase result", 10, "correct", "excellent", "String method"),
    ("Split 'a,b,c' by comma, join with '-'", "Split and join", "print('-'.join('a,b,c'.split(',')))", "a-b-c", "Transformed string", 10, "correct", "excellent", "Split/join"),
    ("Remove whitespace from '  hello  '", "Use strip()", "print('  hello  '.strip())", "hello", "Stripped string", 10, "correct", "excellent", "Strip method"),
    ("Replace spaces with underscores in 'hello world'", "Use replace()", "print('hello world'.replace(' ', '_'))", "hello_world", "Replaced string", 10, "correct", "excellent", "Replace method"),
    ("Check if 'python' starts with 'py'", "Use startswith()", "print('python'.startswith('py'))", "True", "Starts with py", 10, "correct", "excellent", "Built-in method"),
    ("Check if 'file.txt' ends with '.txt'", "Use endswith()", "print('file.txt'.endswith('.txt'))", "True", "Ends with .txt", 10, "correct", "excellent", "Built-in method"),
    ("Title case 'hello world'", "Use title()", "print('hello world'.title())", "Hello World", "Title case", 10, "correct", "excellent", "String method"),
    ("Count vowels in 'programming'", "Check each char", "print(sum(1 for c in 'programming' if c in 'aeiou'))", "3", "3 vowels", 10, "correct", "excellent", "Generator expression"),
    ("Remove duplicate chars from 'aabbcc'", "Use dict.fromkeys", "print(''.join(dict.fromkeys('aabbcc')))", "abc", "Duplicates removed", 10, "correct", "excellent", "Order preserved"),
    ("Capitalize first letter of 'hello'", "Use capitalize()", "print('hello'.capitalize())", "Hello", "Capitalized", 10, "correct", "excellent", "Simple method"),
    ("Check if 'abc123' is alphanumeric", "Use isalnum()", "print('abc123'.isalnum())", "True", "Is alphanumeric", 10, "correct", "excellent", "Built-in check"),
    ("Swap case of 'Hello World'", "Use swapcase()", "print('Hello World'.swapcase())", "hELLO wORLD", "Case swapped", 10, "correct", "excellent", "Swapcase method"),
    ("Find longest word in 'The quick brown fox'", "Split and max", "print(max('The quick brown fox'.split(), key=len))", "quick", "Longest: quick", 10, "correct", "excellent", "Max with key"),
    ("Most common char in 'mississippi'", "Use Counter", "from collections import Counter\nprint(Counter('mississippi').most_common(1)[0])", "('i', 4)", "i appears 4 times", 10, "correct", "excellent", "Counter most_common"),
]

# ============= LIST EXAMPLES (20) =============
LIST_EXAMPLES = [
    ("Sort [3,1,4,1,5,9,2,6]", "Use sorted()", "print(sorted([3,1,4,1,5,9,2,6]))", "[1, 1, 2, 3, 4, 5, 6, 9]", "Sorted list", 10, "correct", "excellent", "Built-in sorted"),
    ("Filter even numbers from 1-10", "List comprehension", "print([x for x in range(1,11) if x % 2 == 0])", "[2, 4, 6, 8, 10]", "Even numbers", 10, "correct", "excellent", "Comprehension"),
    ("Square each element in [1,2,3,4,5]", "List comprehension", "print([x**2 for x in [1,2,3,4,5]])", "[1, 4, 9, 16, 25]", "Squared list", 10, "correct", "excellent", "Comprehension"),
    ("Find min and max in [5,2,8,1,9]", "Use min/max", "nums = [5,2,8,1,9]\nprint(f'min:{min(nums)}, max:{max(nums)}')", "min:1, max:9", "Min=1, Max=9", 10, "correct", "excellent", "Built-in functions"),
    ("Remove duplicates from [1,2,2,3,3,3]", "Use dict.fromkeys", "print(list(dict.fromkeys([1,2,2,3,3,3])))", "[1, 2, 3]", "Duplicates removed", 10, "correct", "excellent", "Order preserved"),
    ("Flatten [[1,2],[3,4],[5,6]]", "Nested comprehension", "print([x for sub in [[1,2],[3,4],[5,6]] for x in sub])", "[1, 2, 3, 4, 5, 6]", "Flattened list", 10, "correct", "excellent", "Nested comprehension"),
    ("Get every 2nd element from [1,2,3,4,5,6,7,8]", "Slice with step", "print([1,2,3,4,5,6,7,8][::2])", "[1, 3, 5, 7]", "Every 2nd element", 10, "correct", "excellent", "Slice syntax"),
    ("Zip [1,2,3] with ['a','b','c']", "Use zip()", "print(list(zip([1,2,3], ['a','b','c'])))", "[(1, 'a'), (2, 'b'), (3, 'c')]", "Zipped pairs", 10, "correct", "excellent", "Built-in zip"),
    ("Sum of [10,20,30,40,50]", "Use sum()", "print(sum([10,20,30,40,50]))", "150", "Sum is 150", 10, "correct", "excellent", "Built-in sum"),
    ("Find index of 30 in [10,20,30,40,50]", "Use index()", "print([10,20,30,40,50].index(30))", "2", "Index is 2", 10, "correct", "excellent", "Index method"),
    ("Reverse [1,2,3,4,5]", "Slice with -1", "print([1,2,3,4,5][::-1])", "[5, 4, 3, 2, 1]", "Reversed list", 10, "correct", "excellent", "Slice reversal"),
    ("Count 3s in [1,3,3,3,5,3,7]", "Use count()", "print([1,3,3,3,5,3,7].count(3))", "4", "4 occurrences", 10, "correct", "excellent", "Count method"),
    ("Get numbers > 5 from [2,4,6,8,10]", "Filter with comprehension", "print([x for x in [2,4,6,8,10] if x > 5])", "[6, 8, 10]", "Filtered list", 10, "correct", "excellent", "Comprehension filter"),
    ("Enumerate ['a','b','c']", "Use enumerate", "print(list(enumerate(['a','b','c'])))", "[(0, 'a'), (1, 'b'), (2, 'c')]", "Enumerated list", 10, "correct", "excellent", "Built-in enumerate"),
    ("Sort tuples by 2nd element", "Sort with key", "print(sorted([('b',2),('a',1),('c',3)], key=lambda x: x[1]))", "[('a', 1), ('b', 2), ('c', 3)]", "Sorted by 2nd", 10, "correct", "excellent", "Lambda key"),
    ("Check if all even in [2,4,6,8]", "Use all()", "print(all(x % 2 == 0 for x in [2,4,6,8]))", "True", "All are even", 10, "correct", "excellent", "All with generator"),
    ("Concatenate [1,2,3] and [4,5,6]", "Use + operator", "print([1,2,3] + [4,5,6])", "[1, 2, 3, 4, 5, 6]", "Concatenated", 10, "correct", "excellent", "List addition"),
    ("Find 2nd largest in [5,2,8,1,9]", "Sort and index", "print(sorted(set([5,2,8,1,9]))[-2])", "8", "2nd largest is 8", 10, "correct", "excellent", "Set removes dups"),
    ("Find common in [1,2,3,4] and [3,4,5,6]", "Set intersection", "print(sorted(set([1,2,3,4]) & set([3,4,5,6])))", "[3, 4]", "Common: [3,4]", 10, "correct", "excellent", "Set intersection"),
    ("Split [1,2,3,4,5,6,7,8] into chunks of 3", "Comprehension with range", "lst = [1,2,3,4,5,6,7,8]\nprint([lst[i:i+3] for i in range(0, len(lst), 3)])", "[[1, 2, 3], [4, 5, 6], [7, 8]]", "Chunked list", 10, "correct", "excellent", "Chunking logic"),
]

# ============= ALGORITHM EXAMPLES (20) =============
ALGO_EXAMPLES = [
    ("Binary search for 7 in [1,3,5,7,9,11]", "Divide search space", "arr = [1,3,5,7,9,11]\nl, r = 0, len(arr)-1\nwhile l <= r:\n    m = (l+r)//2\n    if arr[m] == 7: print(m); break\n    elif arr[m] < 7: l = m+1\n    else: r = m-1", "3", "Found at index 3", 10, "correct", "excellent", "O(log n) search"),
    ("Bubble sort [64,34,25,12,22]", "Swap adjacent elements", "arr = [64,34,25,12,22]\nfor i in range(len(arr)):\n    for j in range(len(arr)-i-1):\n        if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\nprint(arr)", "[12, 22, 25, 34, 64]", "Sorted array", 10, "correct", "good", "O(n^2) sort"),
    ("Factorial of 6 using recursion", "n! = n * (n-1)!", "def f(n): return 1 if n <= 1 else n * f(n-1)\nprint(f(6))", "720", "6! = 720", 10, "correct", "excellent", "Recursive factorial"),
    ("Fibonacci with memoization", "Cache results", "from functools import lru_cache\n@lru_cache\ndef fib(n): return n if n < 2 else fib(n-1) + fib(n-2)\nprint(fib(30))", "832040", "fib(30) = 832040", 10, "correct", "excellent", "Memoized O(n)"),
    ("Two sum: find pair summing to 6 in [1,2,3,4]", "Two pointers", "arr = [1,2,3,4]\nl, r = 0, len(arr)-1\nwhile l < r:\n    s = arr[l] + arr[r]\n    if s == 6: print((arr[l], arr[r])); break\n    elif s < 6: l += 1\n    else: r -= 1", "(2, 4)", "2 + 4 = 6", 10, "correct", "excellent", "O(n) two pointers"),
    ("Max sum subarray of size 3 in [2,1,5,1,3,2]", "Sliding window", "arr = [2,1,5,1,3,2]\nk = 3\nws = sum(arr[:k])\nms = ws\nfor i in range(k, len(arr)):\n    ws = ws - arr[i-k] + arr[i]\n    ms = max(ms, ws)\nprint(ms)", "9", "Max sum is 9", 10, "correct", "excellent", "O(n) sliding window"),
    ("Find 3rd smallest in [7,10,4,3,20,15]", "Sort and index", "print(sorted([7,10,4,3,20,15])[2])", "7", "3rd smallest is 7", 10, "correct", "good", "O(n log n) sort"),
    ("Merge [1,3,5] and [2,4,6]", "Two pointer merge", "a, b = [1,3,5], [2,4,6]\nres = []\ni = j = 0\nwhile i < len(a) and j < len(b):\n    if a[i] <= b[j]: res.append(a[i]); i += 1\n    else: res.append(b[j]); j += 1\nres.extend(a[i:])\nres.extend(b[j:])\nprint(res)", "[1, 2, 3, 4, 5, 6]", "Merged list", 10, "correct", "excellent", "O(n+m) merge"),
    ("Kadane's max subarray for [-2,1,-3,4,-1,2,1,-5,4]", "Track current and global max", "arr = [-2,1,-3,4,-1,2,1,-5,4]\nmc = mg = arr[0]\nfor x in arr[1:]:\n    mc = max(x, mc + x)\n    mg = max(mg, mc)\nprint(mg)", "6", "Max subarray sum is 6", 10, "correct", "excellent", "O(n) Kadane's"),
    ("Rotate [1,2,3,4,5] right by 2", "Slice and concatenate", "arr = [1,2,3,4,5]\nk = 2\nprint(arr[-k:] + arr[:-k])", "[4, 5, 1, 2, 3]", "Rotated array", 10, "correct", "excellent", "O(n) rotation"),
    ("Sort [2,0,1,2,1,0] containing only 0,1,2", "Count and rebuild", "from collections import Counter\narr = [2,0,1,2,1,0]\nc = Counter(arr)\nprint([0]*c[0] + [1]*c[1] + [2]*c[2])", "[0, 0, 1, 1, 2, 2]", "Sorted array", 10, "correct", "excellent", "O(n) counting"),
    ("Find missing number in [3,0,1] (0-3)", "Sum formula", "arr = [3,0,1]\nn = len(arr)\nprint(n*(n+1)//2 - sum(arr))", "2", "Missing is 2", 10, "correct", "excellent", "O(n) math"),
    ("Find duplicate in [1,3,4,2,2]", "Use set", "arr = [1,3,4,2,2]\nseen = set()\nfor x in arr:\n    if x in seen: print(x); break\n    seen.add(x)", "2", "Duplicate is 2", 10, "correct", "excellent", "O(n) with set"),
    ("Majority element in [3,2,3]", "Counter most common", "from collections import Counter\nprint(Counter([3,2,3]).most_common(1)[0][0])", "3", "Majority is 3", 10, "correct", "excellent", "Counter approach"),
    ("Check if 121 is palindrome number", "Reverse and compare", "n = 121\nprint(str(n) == str(n)[::-1])", "True", "121 is palindrome", 10, "correct", "excellent", "String approach"),
    ("Check if 16 is power of 2", "Bit manipulation", "n = 16\nprint(n > 0 and (n & (n-1)) == 0)", "True", "16 is power of 2", 10, "correct", "excellent", "O(1) bit trick"),
    ("Single number in [4,1,2,1,2]", "XOR all", "res = 0\nfor x in [4,1,2,1,2]: res ^= x\nprint(res)", "4", "Single is 4", 10, "correct", "excellent", "O(n) XOR"),
    ("Valid parentheses '([]){}'", "Stack matching", "s = '([]){}'\nstack = []\npairs = {')':'(', ']':'[', '}':'{'}\nvalid = True\nfor c in s:\n    if c in '([{': stack.append(c)\n    elif not stack or stack.pop() != pairs[c]: valid = False; break\nprint(valid and len(stack) == 0)", "True", "Valid parentheses", 10, "correct", "excellent", "Stack approach"),
    ("Move zeros to end in [0,1,0,3,12]", "Two pointer", "arr = [0,1,0,3,12]\npos = 0\nfor i in range(len(arr)):\n    if arr[i] != 0:\n        arr[pos], arr[i] = arr[i], arr[pos]\n        pos += 1\nprint(arr)", "[1, 3, 12, 0, 0]", "Zeros at end", 10, "correct", "excellent", "O(n) in-place"),
    ("Count inversions in [2,4,1,3,5]", "Brute force count", "arr = [2,4,1,3,5]\ncount = sum(1 for i in range(len(arr)) for j in range(i+1, len(arr)) if arr[i] > arr[j])\nprint(count)", "3", "3 inversions", 10, "correct", "good", "O(n^2) brute force"),
]

# ============= DATA STRUCTURE EXAMPLES (20) =============
DS_EXAMPLES = [
    ("Get age from {'name':'Alice','age':25}", "Key access", "print({'name':'Alice','age':25}['age'])", "25", "Age is 25", 10, "correct", "excellent", "Dict access"),
    ("Union and intersection of {1,2,3} and {2,3,4}", "Set operators", "a, b = {1,2,3}, {2,3,4}\nprint(f'union:{a|b}, inter:{a&b}')", "union:{1, 2, 3, 4}, inter:{2, 3}", "Set operations", 10, "correct", "excellent", "Set operators"),
    ("Count chars in 'abracadabra'", "Use Counter", "from collections import Counter\nprint(dict(Counter('abracadabra')))", "{'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}", "Char counts", 10, "correct", "excellent", "Counter usage"),
    ("Stack: push 1,2,3 then pop twice", "List as stack", "s = []\ns.append(1); s.append(2); s.append(3)\ns.pop(); s.pop()\nprint(s)", "[1]", "Stack: [1]", 10, "correct", "excellent", "List stack"),
    ("Queue: enqueue 1,2,3 then dequeue", "Deque", "from collections import deque\nq = deque([1,2,3])\nq.popleft()\nprint(list(q))", "[2, 3]", "Queue: [2,3]", 10, "correct", "excellent", "Deque O(1)"),
    ("Access nested dict {'a':{'b':'value'}}", "Chain keys", "print({'a':{'b':'value'}}['a']['b'])", "value", "Nested access", 10, "correct", "excellent", "Nested dict"),
    ("Group words by first letter", "defaultdict", "from collections import defaultdict\nwords = ['apple','ant','banana']\nd = defaultdict(list)\nfor w in words: d[w[0]].append(w)\nprint(dict(d))", "{'a': ['apple', 'ant'], 'b': ['banana']}", "Grouped words", 10, "correct", "excellent", "defaultdict"),
    ("Dict maintains insertion order", "Python 3.7+", "d = {}\nd['a'] = 1; d['b'] = 2; d['c'] = 3\nprint(list(d.keys()))", "['a', 'b', 'c']", "Order preserved", 10, "correct", "excellent", "Dict ordering"),
    ("Create Point namedtuple", "namedtuple", "from collections import namedtuple\nPoint = namedtuple('Point', ['x','y'])\np = Point(3, 4)\nprint(p.x, p.y)", "3 4", "Point(3,4)", 10, "correct", "excellent", "namedtuple"),
    ("Dict of squares 1-5", "Dict comprehension", "print({x: x**2 for x in range(1,6)})", "{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}", "Squares dict", 10, "correct", "excellent", "Dict comprehension"),
    ("Unique lengths from ['apple','ant','banana']", "Set comprehension", "print(sorted({len(w) for w in ['apple','ant','banana']}))", "[3, 5, 6]", "Unique lengths", 10, "correct", "excellent", "Set comprehension"),
    ("Safe get with default", "dict.get()", "print({'a':1}.get('b', 0))", "0", "Default returned", 10, "correct", "excellent", "Safe access"),
    ("Merge {'a':1} and {'b':2}", "Dict union", "print({'a':1} | {'b':2})", "{'a': 1, 'b': 2}", "Merged dict", 10, "correct", "excellent", "Dict merge"),
    ("3 smallest in [5,2,8,1,9,3,7]", "heapq.nsmallest", "import heapq\nprint(heapq.nsmallest(3, [5,2,8,1,9,3,7]))", "[1, 2, 3]", "3 smallest", 10, "correct", "excellent", "Heap selection"),
    ("Iterate dict items", "items()", "for k,v in {'a':1,'b':2}.items(): print(f'{k}:{v}')", "a:1\nb:2", "Dict iteration", 10, "correct", "excellent", "items() method"),
    ("Set difference {1,2,3,4} - {3,4,5}", "- operator", "print(sorted({1,2,3,4} - {3,4,5}))", "[1, 2]", "Difference: {1,2}", 10, "correct", "excellent", "Set difference"),
    ("ChainMap lookup", "ChainMap", "from collections import ChainMap\ncm = ChainMap({'x':1}, {'y':2})\nprint(cm['x'])", "1", "Found x:1", 10, "correct", "excellent", "ChainMap"),
    ("Top 2 common in [1,1,1,2,2,3]", "Counter", "from collections import Counter\nprint(Counter([1,1,1,2,2,3]).most_common(2))", "[(1, 3), (2, 2)]", "Top 2 elements", 10, "correct", "excellent", "most_common"),
    ("Sort dict by value", "sorted with key", "print(dict(sorted({'c':3,'a':1,'b':2}.items(), key=lambda x: x[1])))", "{'a': 1, 'b': 2, 'c': 3}", "Sorted by value", 10, "correct", "excellent", "Lambda sort"),
    ("Invert dict {'a':1,'b':2}", "Comprehension", "print({v:k for k,v in {'a':1,'b':2}.items()})", "{1: 'a', 2: 'b'}", "Inverted dict", 10, "correct", "excellent", "Dict inversion"),
]

def main():
    print("="*60)
    print("Creating Expanded CodeAct Dataset (100 examples)")
    print("="*60)

    all_examples = []

    # Generate all examples
    categories = [
        ("Math", MATH_EXAMPLES),
        ("String", STRING_EXAMPLES),
        ("List", LIST_EXAMPLES),
        ("Algorithm", ALGO_EXAMPLES),
        ("Data Structure", DS_EXAMPLES),
    ]

    for name, examples in categories:
        print(f"Adding {name} examples ({len(examples)})...")
        for ex in examples:
            all_examples.append(create_example(*ex))

    # Save to file
    os.makedirs("data", exist_ok=True)
    output_file = "data/codeact_feedback_train_100.jsonl"

    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    file_size = os.path.getsize(output_file) / 1024

    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total examples: {len(all_examples)}")
    print(f"  - Math: 20")
    print(f"  - String: 20")
    print(f"  - List: 20")
    print(f"  - Algorithm: 20")
    print(f"  - Data Structure: 20")
    print(f"\nSaved to: {output_file}")
    print(f"File size: {file_size:.2f} KB")

def create_correction_example(task, wrong_thought, wrong_code, wrong_output, error_feedback, correct_thought, correct_code, correct_output, solution):
    """Create an example showing error correction"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"<thought>{wrong_thought}</thought>\n\n<execute>\n{wrong_code}\n</execute>"},
            {"role": "user", "content": f"Execution result: {wrong_output}\n\nERROR: {error_feedback}"},
            {"role": "assistant", "content": f"<thought>I made a mistake. {correct_thought}</thought>\n\n<execute>\n{correct_code}\n</execute>\n\n<solution>{solution}</solution>"}
        ]
    }

# ============= ERROR CORRECTION EXAMPLES (10) =============
CORRECTION_EXAMPLES = [
    # Wrong: incomplete sieve, Correct: complete sieve
    ("Find the first 10 prime numbers",
     "Use sieve of eratosthenes",
     "n = 30\na = [True] * n\na[0], a[1] = False, False\nprint([i for i in range(2, n) if a[i]])",
     "[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]",
     "This includes non-primes like 4, 6, 8, 9. The sieve is incomplete - you didn't mark composites!",
     "I forgot to mark composite numbers. Need to iterate and mark multiples.",
     "def first_n_primes(n):\n    primes = []\n    num = 2\n    while len(primes) < n:\n        is_prime = all(num % p != 0 for p in primes)\n        if is_prime:\n            primes.append(num)\n        num += 1\n    return primes\nprint(first_n_primes(10))",
     "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]",
     "First 10 primes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]"),

    # Wrong: wrong count, Correct: exact count
    ("Count the letter 'r' in 'raoanvdsknjrrrr'",
     "Count occurrences of r",
     "s = 'raoanvdsknjrrrr'\nprint(len(s))",
     "15",
     "That's the string length, not the count of 'r'. I asked for how many 'r' letters.",
     "I printed length instead of counting 'r'. Need to use count() method.",
     "s = 'raoanvdsknjrrrr'\nprint(s.count('r'))",
     "5",
     "The letter 'r' appears 5 times"),

    # Wrong: sum instead of product
    ("Calculate the product of numbers 1 to 5",
     "Multiply 1 through 5",
     "print(sum(range(1, 6)))",
     "15",
     "That's the SUM (15), not the PRODUCT (120). 1*2*3*4*5 = 120",
     "I used sum() instead of product. Need to multiply all numbers.",
     "from functools import reduce\nprint(reduce(lambda x,y: x*y, range(1, 6)))",
     "120",
     "Product of 1 to 5 is 120"),

    # Wrong: unsorted, Correct: sorted
    ("Sort [5, 2, 8, 1, 9] in descending order",
     "Sort the list",
     "print(sorted([5, 2, 8, 1, 9]))",
     "[1, 2, 5, 8, 9]",
     "This is ascending order! I asked for DESCENDING (largest first).",
     "I forgot reverse=True for descending order.",
     "print(sorted([5, 2, 8, 1, 9], reverse=True))",
     "[9, 8, 5, 2, 1]",
     "Descending order: [9, 8, 5, 2, 1]"),

    # Wrong: off by one
    ("Find the 5th Fibonacci number (0, 1, 1, 2, 3, 5, ...)",
     "Calculate 5th fibonacci",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(5))",
     "5",
     "The 5th Fibonacci is 3, not 5. Sequence: F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5",
     "Off-by-one error. F(5)=5 but F(4)=3. User wanted index 4 (5th number counting from 0).",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(4))  # 5th number is at index 4",
     "3",
     "5th Fibonacci number is 3"),
]

def main_with_corrections():
    """Main function that includes correction examples"""
    print("="*60)
    print("Creating Expanded CodeAct Dataset (100 + 10 correction examples)")
    print("="*60)

    all_examples = []

    categories = [
        ("Math", MATH_EXAMPLES),
        ("String", STRING_EXAMPLES),
        ("List", LIST_EXAMPLES),
        ("Algorithm", ALGO_EXAMPLES),
        ("Data Structure", DS_EXAMPLES),
    ]

    for name, examples in categories:
        print(f"Adding {name} examples ({len(examples)})...")
        for ex in examples:
            all_examples.append(create_example(*ex))

    # Add correction examples
    print(f"Adding Correction examples ({len(CORRECTION_EXAMPLES)})...")
    for ex in CORRECTION_EXAMPLES:
        all_examples.append(create_correction_example(*ex))

    os.makedirs("data", exist_ok=True)
    output_file = "data/codeact_feedback_train_100.jsonl"

    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    file_size = os.path.getsize(output_file) / 1024

    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total examples: {len(all_examples)}")
    print(f"  - Regular examples: 100")
    print(f"  - Correction examples: {len(CORRECTION_EXAMPLES)}")
    print(f"\nSaved to: {output_file}")
    print(f"File size: {file_size:.2f} KB")

if __name__ == "__main__":
    main_with_corrections()
