# Handoff notes

Context for continuing this project in a new Claude Code session.

## Current state (2026-10-07)

**Data:** (Gutenberg header and license stripped from the Alice file on 2026-10-08; original not kept in the project) `data/books` holds `alice_in_wonderland.md` and `The-Complete-Guide-to-Trading.pdf` (116 pages, CFI).
The pod assignment file was removed from the project (copies remain in Downloads and the old OneDrive folder).

**Pipeline** (`query_data.py`, `MODE` switch):
- `vector`: Chroma similarity search, top 4, guardrail at relevance 0.3 (the original)
- `hybrid`: vector top 20 + BM25 top 20 (`rank_bm25`, Porter-stemmed tokens), merged with Reciprocal Rank Fusion
- `rerank` (default): hybrid candidates re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` (sigmoid to 0-1), top 4, guardrail at 0.5
- LLM is `llama3.2` at `temperature=0` (set 2026-10-08) so answers repeat across runs; before that, the Mad Hatter question flipped between a correct answer and "I don't know."

**Ingestion** (`create_database.py`): PDF lines repeated on more than half the pages (headers/footers) and bare page numbers are dropped.

**Testing:**
- `tests.json`: 16 answerable questions (14 trading PDF, 2 Alice) + 4 off-topic questions that should be refused
- `run_tests.py [vector|hybrid|rerank|all]`: string-match pass/fail for retrieval and answer
- `evaluate.py [mode|all]`: Ragas faithfulness, context recall, answer relevancy with local llama3.2 as judge; per-question CSVs in `eval_results/`

## Results so far (run_tests.py all)

| Mode | Retrieval | Answers |
|---|---|---|
| vector | 15/16 | 16/20 |
| hybrid | 15/16 | 16/20 |
| rerank | 14/16 | 18/20 |

- The rerank guardrail separates cleanly: real questions score 0.95-1.00, off-topic ones 0.00-0.07. The old 0.3 vector cutoff blocked 2 real questions.
- Hybrid ties vector on this test set; its benefit is for exact terms (codes, names, numbers) that meaning search misses.
- Known misses: "copyrighted" year question (the word "copyright" also appears 10 times in Alice's Project Gutenberg license); raven riddle drops to 5th after re-ranking because its passage doesn't contain the word "riddle".

## Gotchas

- **Ragas 0.4.3 + langchain-community 0.4** are incompatible (`No module named 'langchain_community.chat_models.vertexai'`). `evaluate.py` registers a placeholder module before importing ragas.
- **Ragas is slow** with a local judge: about 1 minute per metric per question.
- First run of a new Hugging Face model can stall on download; `HF_HUB_OFFLINE=1` skips the network once models are cached.
- The OneDrive copy's `.venv` was broken by the company security scanner quarantining files; keep the project outside OneDrive.

## Changes on 2026-10-08

- `temperature=0`, `K=5`, `.docx` loader (`python-docx`), Ragas judge switched to `llama3.1:8b` with `max_workers=1`.
- Rebuilt DB: 463 chunks. `run_tests.py` rerank: retrieval 14/16, answers 18/20.
- First Ragas run (llama3.2 judge, before these changes): faithfulness 0.93, context recall 0.78, answer relevancy unusable (13/16 blank).
- Stripping the license did NOT fix the copyright question: the cross-encoder ranks the "Copyright 2018 CFI" chunk 10th, below the title pages (top 0.497 < 0.5 cutoff).
- K=5 did NOT fix the riddle: after the rebuild the raven chunk ranks 6th; tea-party chunks are all scored 0.98-1.00.
- Claude Code's background jobs get killed on low memory; run `evaluate.py` from your own terminal.

## Next steps

1. Run `python evaluate.py` from your own terminal (llama3.1:8b judge) and record the scores in README.
2. Decide on the two remaining misses: larger chunks (riddle) and/or a stronger re-ranker such as `BAAI/bge-reranker-base` (copyright).
3. Grow `tests.json` to 30+ questions, including some with exact codes/numbers where BM25 should beat vector.
4. Optional: a Streamlit UI.
