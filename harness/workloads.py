"""Fixed workload set. Every workload is a chat request with fixed prompt, seed, greedy sampling
and a fixed output cap, so runs are comparable across engines/configs.

  chat_short   short chat turn                         (~40 prompt tok, <=128 out)
  long_prompt  ~3.3k-token context, short answer        (prompt processing dominates)
  long_gen     long free-form generation                (<=400 out)
  code_edit    rewrite a code file with small changes   (repetitive output; best case for drafting, <=640 out)
  extract      structured JSON extraction from text     (<=448 out)
"""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI = os.path.join(ROOT, "data", "wikitext-2-raw", "wiki.valid.raw")

SYSTEM = "You are a helpful assistant."


def wiki_articles():
    t = open(WIKI, encoding="utf-8").read()
    parts = re.split(r"\n = ([^=\n]+) = \n", t)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def long_context():
    a = wiki_articles()
    names = ["Homarus gammarus", "M @-@ 82 ( Michigan highway )"]
    return "\n\n".join(f"# {n}\n\n{a[n]}" for n in names)


CODE_FILE = '''import csv
import json


def load_orders(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def total_by_customer(rows):
    totals = {}
    for row in rows:
        name = row["customer"]
        amount = float(row["amount"])
        if name not in totals:
            totals[name] = 0.0
        totals[name] += amount
    return totals


def top_customers(totals, n):
    items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return items[:n]


def filter_by_status(rows, status):
    out = []
    for row in rows:
        if row["status"] == status:
            out.append(row)
    return out


def average_order_value(rows):
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += float(row["amount"])
    return total / len(rows)


def save_report(path, totals, top):
    report = {"totals": totals, "top": top}
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    rows = load_orders("orders.csv")
    paid = filter_by_status(rows, "paid")
    totals = total_by_customer(paid)
    top = top_customers(totals, 5)
    print("average order value:", average_order_value(paid))
    save_report("report.json", totals, top)


if __name__ == "__main__":
    main()
'''

EXTRACT_TEXT = """Shipment notes, week 14.
On Monday 3 April, Halvorsen Freight delivered 240 crates of ceramic tiles to the Rotterdam depot; the invoice came to 18,450.00 EUR and was signed for by M. de Vries.
Two days later, on 5 April, Brightline Logistics brought 75 pallets of copper wire to Antwerp. Invoice total: 62,300.50 EUR. Receiver: A. Peeters.
Kestrel Cargo arrived at the Hamburg depot on 6 April with 1,200 boxes of LED fixtures, invoiced at 27,980.00 EUR and received by J. Brandt.
On Friday 7 April, Halvorsen Freight returned to Rotterdam with 60 drums of industrial solvent (invoice 9,120.75 EUR), again signed by M. de Vries.
Finally, Nordsee Transport delivered 430 sacks of cement to Bremen on 8 April for 5,675.00 EUR; S. Köhler signed."""


def workloads():
    ctx = long_context()
    return {
        "chat_short": {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Explain in three or four sentences why the sky is blue during the day but red at sunset."}],
            "max_tokens": 128,
        },
        "long_prompt": {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Read the following two articles and then answer the question.\n\n" + ctx +
                          "\n\nQuestion: According to the articles, (1) what colour are European lobsters in life, and (2) between which two places does M-82 travel? Answer in one short sentence each."}],
            "max_tokens": 64,
        },
        "long_gen": {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Write a detailed essay of about 600 words on the history and impact of the printing press, from Gutenberg to the modern era."}],
            "max_tokens": 400,
        },
        "code_edit": {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Here is a Python file. Add type hints to every function signature and a one-line docstring to every function. Do not change any other code. Output the complete updated file in a single ```python code block and nothing else.\n\n```python\n" + CODE_FILE + "```"}],
            "max_tokens": 640,
        },
        "extract": {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Extract every shipment from the notes below as a JSON array. Each element must have exactly these keys: \"date\" (YYYY-MM-DD, year 2023), \"carrier\", \"depot\", \"quantity\" (integer), \"unit\", \"goods\", \"invoice_eur\" (number), \"receiver\". Output only the JSON.\n\n" + EXTRACT_TEXT}],
            "max_tokens": 448,
        },
    }


if __name__ == "__main__":
    for k, w in workloads().items():
        print(k, sum(len(m["content"]) for m in w["messages"]), "chars, max_tokens", w["max_tokens"])
