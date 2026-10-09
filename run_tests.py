"""Run the test questions in tests.json and report retrieval and answer accuracy.

    python run_tests.py            # default mode (rerank)
    python run_tests.py vector     # one mode: vector, hybrid or rerank
    python run_tests.py all        # all three, with a comparison table at the end

retrieval: every expected string appears in the passages sent to the LLM (did search find the right passage?)
answer:    every expected string appears in the LLM's answer (did the model use it correctly?)
           For questions with no expected strings, the right answer is a refusal.
"""
import json
import sys
import time
from query_data import ask, MODE

TESTS = json.load(open("tests.json", encoding="utf-8"))
REFUSALS = ["no good match found", "i don't know", "i do not know"]


def normalize(text):
    return text.lower().replace("’", "'")


def contains_all(text, expected):
    return all(normalize(e) in normalize(text) for e in expected)


def run(mode):
    rows = []
    for test in TESTS:
        question, expected = test["question"], test["must_contain"]
        start = time.time()
        answer, results = ask(question, mode)
        seconds = time.time() - start
        context = " ".join(doc.page_content for doc, _ in results)
        top_score = results[0][1] if results else 0

        if expected:
            retrieved = contains_all(context, expected)
            answered = contains_all(answer, expected)
        else:
            retrieved = None  # nothing to find
            answered = any(r in normalize(answer) for r in REFUSALS)
        rows.append((question, retrieved, answered, top_score, seconds, answer))

    mark = {True: "PASS", False: "FAIL", None: " -- "}
    print(f"\n=== mode: {mode} ===")
    print(f"{'retrieval':9}  {'answer':6}  {'top':5}  {'secs':4}  question")
    for question, retrieved, answered, top_score, seconds, _ in rows:
        print(f"{mark[retrieved]:9}  {mark[answered]:6}  {top_score:5.2f}  {seconds:4.1f}  {question}")

    retrieval_rows = [r for r in rows if r[1] is not None]
    summary = (sum(r[1] for r in retrieval_rows), len(retrieval_rows), sum(r[2] for r in rows), len(rows))
    print(f"\nRetrieval: {summary[0]}/{summary[1]}   Answers: {summary[2]}/{summary[3]}")

    print("\nFailed answers:")
    for question, _, answered, _, _, answer in rows:
        if not answered:
            print(f"\n> {question}\n{answer.strip()[:300]}")
    return summary


modes = ["vector", "hybrid", "rerank"] if sys.argv[1:] == ["all"] else sys.argv[1:] or [MODE]
summaries = {mode: run(mode) for mode in modes}

if len(modes) > 1:
    print("\n=== comparison ===")
    for mode, (r_ok, r_total, a_ok, a_total) in summaries.items():
        print(f"{mode:7}  retrieval {r_ok}/{r_total}   answers {a_ok}/{a_total}")
