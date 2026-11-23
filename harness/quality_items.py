"""48 fixed task items with mechanically checkable answers.

Each item: id, category, prompt, answer (string or list of acceptable strings), kind
  kind = "num"   -> last number in the reply's final "Answer:" line must equal answer
  kind = "text"  -> any acceptable string appears (case-insensitive) in the final "Answer:" line
Categories: arith (direct computation), reason (multi-step word/logic problems; where thinking should
help), fact (recall), read (answer is in a supplied passage), code (predict program output).
"""

SUFFIX = "\n\nFinish your reply with a final line of the form `Answer: <answer>`."

PASSAGE = (
    "The Tolland Canal was authorised by Parliament in 1794 and opened in 1801. It ran 23 miles from the "
    "market town of Abbotsley to the port of Wexmouth and had 17 locks. Its chief engineer, Margaret Ellison, "
    "was succeeded in 1799 by Thomas Rudge after a dispute over costs, which had risen from an estimated "
    "£84,000 to £131,500. The main cargo was limestone, followed by coal and timber. Traffic peaked in 1838 "
    "at 212,000 tons, declined after the Abbotsley and Wexmouth Railway opened in 1846, and the canal closed "
    "to commercial traffic in 1921. A restoration trust formed in 1974 reopened the first nine miles in 1989."
)

CODE = [
    ("code1", "x = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(sum(v for v in x if v % 2 == 1))", "19"),
    ("code2", "s = 'benchmark'\nprint(s[::-1][2:5])", "amh"),
    ("code3", "d = {}\nfor w in 'the cat and the hat and the bat'.split():\n    d[w] = d.get(w, 0) + 1\nprint(d['the'] * 10 + d['and'])", "32"),
    ("code4", "def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\nprint(f(10))", "55"),
    ("code5", "a = [i * i for i in range(6)]\nprint(a[-2] - a[2])", "12"),
    ("code6", "n = 0\nfor i in range(1, 20):\n    if i % 3 == 0 or i % 5 == 0:\n        n += 1\nprint(n)", "8"),
    ("code7", "x = 17\ny = x // 5\nz = x % 5\nprint(y * 100 + z)", "302"),
    ("code8", "words = ['pear', 'fig', 'banana', 'kiwi']\nprint(max(words, key=len).upper()[:3])", "BAN"),
]

ITEMS = [
    # arithmetic
    ("arith1", "arith", "What is 47 * 23?", "1081", "num"),
    ("arith2", "arith", "What is 1234 + 5678?", "6912", "num"),
    ("arith3", "arith", "What is 9000 - 3457?", "5543", "num"),
    ("arith4", "arith", "What is 144 / 12?", "12", "num"),
    ("arith5", "arith", "What is 15% of 240?", "36", "num"),
    ("arith6", "arith", "What is 2 to the power of 11?", "2048", "num"),
    ("arith7", "arith", "What is 37 * 41?", "1517", "num"),
    ("arith8", "arith", "What is the remainder when 1000 is divided by 7?", "6", "num"),
    ("arith9", "arith", "What is 3/8 written as a decimal?", "0.375", "num"),
    ("arith10", "arith", "What is 125 * 16?", "2000", "num"),
    # multi-step reasoning
    ("reason1", "reason", "A shop sells pens at 3 for $2. How many dollars do 18 pens cost?", "12", "num"),
    ("reason2", "reason", "A train leaves at 09:40 and arrives at 13:15 the same day. How many minutes does the journey take?", "215", "num"),
    ("reason3", "reason", "Tom is twice as old as Ann. In 6 years the sum of their ages will be 48. How old is Ann now?", "12", "num"),
    ("reason4", "reason", "A rectangle has a perimeter of 46 cm and its length is 5 cm more than its width. What is its area in square centimetres?", "126", "num"),
    ("reason5", "reason", "I have 3 boxes. Each box holds 4 bags. Each bag holds 6 marbles, except that one bag in total is empty. How many marbles do I have?", "66", "num"),
    ("reason6", "reason", "A tank is filled by pipe A in 6 hours and by pipe B in 3 hours. How many hours do both pipes together take to fill it?", "2", "num"),
    ("reason7", "reason", "Five people finish a race. Ada finishes before Ben. Cy finishes after Ben but before Dee. Eve finishes before Ada. Who finishes third?", "Ben", "text"),
    ("reason8", "reason", "What is the next number in the sequence 2, 6, 12, 20, 30, ...?", "42", "num"),
    ("reason9", "reason", "A jacket costs $80 after a 20% discount. What was the original price in dollars?", "100", "num"),
    ("reason10", "reason", "How many times does the digit 7 appear in the whole numbers from 1 to 100?", "20", "num"),
    ("reason11", "reason", "If today is Wednesday, what day of the week will it be 100 days from now?", "Friday", "text"),
    ("reason12", "reason", "A car travels 150 km at 50 km/h and then 150 km at 75 km/h. What is its average speed for the whole trip in km/h?", "60", "num"),
    ("reason13", "reason", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost?", "5", "num"),
    ("reason14", "reason", "How many letters 'r' are in the word 'strawberry'?", "3", "num"),
    # factual recall
    ("fact1", "fact", "What is the chemical symbol for gold?", "Au", "text"),
    ("fact2", "fact", "What is the capital city of Australia?", "Canberra", "text"),
    ("fact3", "fact", "In which year did the Second World War end?", "1945", "num"),
    ("fact4", "fact", "Which planet in our solar system has the most mass?", "Jupiter", "text"),
    ("fact5", "fact", "Who wrote the play 'Hamlet'?", "Shakespeare", "text"),
    ("fact6", "fact", "What is the atomic number of carbon?", "6", "num"),
    ("fact7", "fact", "What is the largest ocean on Earth?", "Pacific", "text"),
    ("fact8", "fact", "How many sides does a hexagon have?", "6", "num"),
    ("fact9", "fact", "What is the boiling point of water at sea level in degrees Fahrenheit?", "212", "num"),
    ("fact10", "fact", "Which element has the chemical symbol 'Na'?", "Sodium", "text"),
    # reading comprehension over a passage
    ("read1", "read", PASSAGE + "\n\nIn what year did the canal open?", "1801", "num"),
    ("read2", "read", PASSAGE + "\n\nHow many locks did the canal have?", "17", "num"),
    ("read3", "read", PASSAGE + "\n\nWho was chief engineer when the canal opened? Give the surname.", "Rudge", "text"),
    ("read4", "read", PASSAGE + "\n\nBy how many pounds did the final cost exceed the estimate?", "47500", "num"),
    ("read5", "read", PASSAGE + "\n\nHow many years passed between the railway opening and the canal closing to commercial traffic?", "75", "num"),
    ("read6", "read", PASSAGE + "\n\nWhat was the canal's main cargo?", "limestone", "text"),
    ("read7", "read", PASSAGE + "\n\nHow many miles of the canal remained unrestored after the 1989 reopening?", "14", "num"),
    ("read8", "read", PASSAGE + "\n\nIn which town did the canal start?", "Abbotsley", "text"),
] + [(i, "code", "What does this Python program print?\n\n```python\n" + src + "\n```", ans, "text" if not ans.isdigit() else "num")
     for i, src, ans in CODE]


def items():
    return [{"id": i, "category": c, "prompt": p + SUFFIX, "answer": a, "kind": k} for i, c, p, a, k in ITEMS]


if __name__ == "__main__":
    print(len(items()), "items")
