"""Score the pipeline with Ragas, using local Llama 3.2 as the judge (no API key, no cost).

    python evaluate.py             # default mode (rerank)
    python evaluate.py all         # vector, hybrid and rerank, with a comparison table

Metrics (0 to 1, higher is better):
  faithfulness      Is every claim in the answer backed by the retrieved passages? (catches made-up answers)
  context_recall    Do the retrieved passages contain the facts in the reference answer? (catches search misses)
  answer_relevancy  Does the answer actually address the question?

Only questions that have an answer are scored here; refusals are checked by run_tests.py.
Per-question scores are saved to eval_results/<mode>.csv.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ragas 0.4.3 imports ChatVertexAI from a module that langchain-community 0.4 removed.
# Ragas only uses it in an isinstance() check, so a placeholder class is enough.
import types
_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai)

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from langchain_ollama import ChatOllama
from query_data import ask, embeddings, MODE

TESTS = [t for t in json.load(open("tests.json", encoding="utf-8")) if t["must_contain"]]
# Grader: larger free local model. llama3.2 (3B) often returned malformed output, leaving 21 of 48 scores blank.
# Needs `ollama pull llama3.1:8b` (~5 GB) and more memory; close heavy apps before running.
JUDGE = ChatOllama(model="llama3.1:8b", temperature=0)
METRICS = ["faithfulness", "context_recall", "answer_relevancy"]


def evaluate_mode(mode):
    samples = []
    for test in TESTS:
        answer, results = ask(test["question"], mode)
        samples.append({
            "user_input": test["question"],
            "response": answer,
            "retrieved_contexts": [doc.page_content for doc, _ in results],
            "reference": test["reference"],
        })
        print(f"  [{mode}] answered {len(samples)}/{len(TESTS)}", flush=True)

    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), LLMContextRecall(), ResponseRelevancy()],
        llm=JUDGE,
        embeddings=embeddings,
        run_config=RunConfig(timeout=600, max_workers=1),  # local Ollama with an 8B judge: one request at a time to save memory
    )
    table = result.to_pandas()
    Path("eval_results").mkdir(exist_ok=True)
    table.to_csv(f"eval_results/{mode}.csv", index=False)

    # A judge reply that can't be parsed scores NaN; report how many so averages aren't misleading
    scores = {}
    for metric in METRICS:
        column = table[metric]
        scores[metric] = (column.mean(), column.isna().sum())
    return scores


modes = ["vector", "hybrid", "rerank"] if sys.argv[1:] == ["all"] else sys.argv[1:] or [MODE]
all_scores = {mode: evaluate_mode(mode) for mode in modes}

print(f"\n{'mode':8}" + "".join(f"{m:>20}" for m in METRICS))
for mode, scores in all_scores.items():
    cells = [f"{mean:.2f}" + (f" ({failed} failed)" if failed else "") for mean, failed in scores.values()]
    print(f"{mode:8}" + "".join(f"{c:>20}" for c in cells))
print(f"\n{len(TESTS)} questions per mode. Details in eval_results/")
