# Local RAG: Ask Questions About a Book

A simple **Retrieval-Augmented Generation (RAG)** project that answers questions about *Alice's Adventures in Wonderland*. Everything runs **on your own computer**: no API key, no cost, and no data leaves your machine.

```
python query_data.py "How does Alice meet the Mad Hatter?"

Alice meets the Mad Hatter at the March Hare's house, where they are having a tea party...
Sources: [71809, 73334, 75887, 76751]
```

---

## What is RAG?

Think of a **librarian**:

1. **Before anyone asks anything**, the librarian cuts the book into short passages and files each one by its *meaning*.
2. **When you ask a question**, the librarian pulls out the 4 passages closest in meaning to your question.
3. **The librarian hands those passages to a writer** (the AI) and says: "Answer using only these pages."

**R**etrieve the right pages, then **G**enerate an answer from them. The AI doesn't need to have memorized the book.

```
ONCE:      Book ──► cut into passages ──► meaning-numbers ──► saved in chroma/

EACH TIME: Question ──┬─► meaning search  (top 20) ─┐
                      └─► keyword search  (top 20) ─┴─► merge ──► re-rank ──► best 4
                                                                                │
                                              confident enough? ── no ──► "No good match found"
                                                    │ yes
                                                    ▼
                      Llama 3.2: "answer using only these pages" ──► Answer + Sources
```

---

## Project structure

```
RAG/
├── .venv/                  # Python virtual environment (installed libraries; don't edit)
├── data/
│   └── books/
│       ├── alice_in_wonderland.md            # source files (.md, .txt, .pdf; subfolders OK)
│       └── The-Complete-Guide-to-Trading.pdf
├── chroma/                 # vector database (created by create_database.py)
├── eval_results/           # per-question Ragas scores (created by evaluate.py)
├── create_database.py      # Step 1: build the database (run once)
├── query_data.py           # Step 2: ask questions (run anytime)
├── tests.json              # test questions with expected facts and reference answers
├── run_tests.py            # quick pass/fail check of every test question
├── evaluate.py             # Ragas scores: faithfulness, context recall, answer relevancy
└── README.md
```

---

## Tools used

| Tool | Role (restaurant analogy) | What it does here |
|---|---|---|
| **LangChain** | The manager | Glue library that connects all the parts through one common interface |
| **Hugging Face** (`all-MiniLM-L6-v2`) | The ingredient supplier | Turns text into 384 meaning-numbers (embeddings) |
| **Chroma** | The pantry | Local vector database that stores chunks and their numbers |
| **Ollama** | The kitchen | Runs AI models locally (a background server at `localhost:11434`) |
| **Llama 3.2** (by Meta) | The chef | The language model that writes the final answer (about 3B parameters, about 2 GB) |
| **BM25** (`rank_bm25` + NLTK stemmer) | The index at the back of a book | Keyword search: finds passages containing the question's exact words (names, numbers, codes) |
| **Cross-encoder** (`ms-marco-MiniLM-L-6-v2`) | The taste tester | Reads the question and each candidate passage together and scores how well it answers it |
| **Ragas** | The food critic | Grades answers: grounded in the passages? passages complete? on-topic? |

---

## Setup (one time)

**1. Create and activate a virtual environment** (from the `RAG` folder):
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```
You should now see `(.venv)` at the start of your prompt.

**2. Install the Python libraries:**
```powershell
pip install langchain langchain-text-splitters langchain-chroma langchain-huggingface langchain-ollama sentence-transformers pypdf python-docx rank_bm25 nltk ragas
```

**3. Install Ollama and download the model:**
- Download Ollama from https://ollama.com/download/windows and install it.
- Then run:
```powershell
ollama pull llama3.2
ollama pull llama3.1:8b   # only for evaluate.py (the grader, ~5 GB)
```

---

## How to run

Always run from the `RAG` folder with `(.venv)` active.

**Build the database** (run once, and again whenever you add or change books):
```powershell
python create_database.py
```
```
Loaded alice_in_wonderland.md: 1 document(s)
Loaded The-Complete-Guide-to-Trading.pdf: 116 document(s)
Split 117 document(s) into 491 chunks
Saved to chroma/
```

**Ask questions:**
```powershell
python query_data.py "How does Alice meet the Mad Hatter?"
python query_data.py "Who developed the TRIN indicator?"   # PDF → sources show page numbers
python query_data.py "Who is Harry Potter?"     # not in the books → "No good match found"
```

**Test it** (rerun after every change):
```powershell
python run_tests.py          # pass/fail on every question in tests.json (about 2 min)
python run_tests.py all      # compare vector vs. hybrid vs. rerank
python evaluate.py           # Ragas scores (slow: the local LLM is also the grader)
python evaluate.py all       # Ragas scores for all three modes
```
The LLM runs at `temperature=0`, so the same question gets the same answer every run and a pass/fail result is repeatable. `run_tests.py` checks for exact words (e.g. "tea"), so a correct answer worded differently can still show FAIL; Ragas judges meaning instead.

### Latest results (`run_tests.py`, rerank mode, `temperature=0`, `K=5`)

**Retrieval 14/16, answers 18/20.** All 13 trading questions and the Mad Hatter question passed; all 4 off-topic questions were refused (top score 0.00–0.07 vs. 0.95–1.00 for real questions).

| Failed question | Why |
|---|---|
| What riddle does the Hatter ask Alice? | The riddle line ("Why is a raven like a writing-desk?") sits in a passage without the word "riddle". The re-ranker scores the tea-party passages 0.98–1.00, nearly tied, and puts it 6th |
| In what year was the trading guide copyrighted? | The re-ranker puts the copyright page 10th (it rates the title pages higher), so the best score is 0.497, just under the 0.5 cutoff. Removing the Gutenberg license did not change this |

**Ragas** (`evaluate.py`, rerank mode, `llama3.2` judge, before the changes above): faithfulness **0.93**, context recall **0.78**. Answer relevancy is not reported: the 3B judge returned malformed output for 13 of 16 questions (21 of 48 scores blank overall). The judge is now `llama3.1:8b`; rerun to get complete scores.

---|---|
| How does Alice meet the Mad Hatter? | The answer was correct but didn't use the word "tea" that the test looks for |
| What riddle does the Hatter ask Alice? | The riddle passage doesn't contain the word "riddle", so re-ranking put it 5th and only the top 4 reach the LLM |
| In what year was the trading guide copyrighted? | The Project Gutenberg license in the Alice file repeats "copyright", crowding out the right page (best score 0.50, at the cutoff) |

Ragas (`evaluate.py`) has not been run yet. Expect about 1 hour per mode.

---

## How it works

### `create_database.py`: filing the book

| Step | Code | In plain words |
|---|---|---|
| 1. Load | `LOADERS[path.suffix]` | Find every `.md`, `.txt`, `.pdf` and `.docx` in `data/books` (and subfolders) and hand each to the loader for its type. Text files are read whole; PDFs are read page by page with `pypdf`, and each page remembers its `page` number. Lines repeated on most pages (headers, footers) and bare page numbers are dropped, since they are noise for search. Word files are read with `python-docx` (paragraphs and table rows). Other file types are skipped. |
| 2. Split | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` | Cut the book into passages of about 1,000 characters. Neighbors overlap by 200 characters so no sentence is lost at a cut. `add_start_index` records where each passage starts (these become the "Sources" numbers). |
| 3. Embed | `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` | Turn each passage into 384 numbers that capture its meaning |
| 4. Store | `Chroma.from_documents(...)` | Save the passages and their numbers in the `chroma/` folder (the old one is deleted first) |

### `query_data.py`: answering a question

| Step | Code | In plain words |
|---|---|---|
| 1. Prompt template | `ChatPromptTemplate` | An instruction sheet: "Answer only from this context; otherwise say I don't know" |
| 2a. Meaning search | `similarity_search_with_relevance_scores(question, k=20)` | Turn the question into 384 numbers and find the 20 passages closest in *meaning* |
| 2b. Keyword search | `bm25_search(question, 20)` | Find the 20 passages that share the most *words* with the question, weighting rare words (like "TRIN") above common ones (like "the") |
| 3. Merge | `fuse(...)` (Reciprocal Rank Fusion) | Combine both lists. A passage near the top of both lists ranks highest. Only positions matter, so the two very different scores never need to be compared |
| 4. Re-rank | `rerank(question, docs)` | The cross-encoder reads the question and each passage *together* and scores 0–1 how well it answers. Keep the best 5 |
| 5. Safety check | `best score < 0.5` | If even the best passage is a poor answer, stop instead of letting the AI guess |
| 6. Build context | `"\n\n---\n\n".join(...)` | Join the 5 passages into one block of text |
| 7. Generate | `ChatOllama("llama3.2", temperature=0).invoke(...)` | Send the filled-in prompt to Llama 3.2 and get the answer |
| 8. Show | `print(...)` | Print the answer, the file and page of each passage, and its score |

`MODE` at the top of `query_data.py` switches the pipeline: `"vector"` (steps 2a only, the original), `"hybrid"` (2a + 2b + 3) or `"rerank"` (everything, the default).

---

## Key concepts

### Chunking vs. tokenizing vs. embedding

| | Chunking | Tokenizing | Embedding |
|---|---|---|---|
| **Done by** | LangChain text splitter | Hugging Face model's tokenizer | Hugging Face model |
| **What it does** | Cuts the book into passages of about 1,000 characters | Chops text into word pieces and gives each an ID | Turns the pieces into meaning-numbers |
| **Intelligent?** | No (just cutting) | No (dictionary lookup) | Yes (trained neural network) |
| **Example output** | `"...Alice sat down at the tea table..."` | `["who", "is", "alt", "##ime", ...]` → `[2040, 2003, 2632, ...]` | `[0.12, -0.45, 0.88, ...]` (384 numbers) |

### How embedding works (3 stages)

1. **Look up:** each token ID gets a starting set of 384 numbers from a table the model learned during training.
2. **Attention ("talking to each other"):** across 6 layers, every piece adjusts its numbers based on the others. That's how "bank" means something different in "river bank" and "bank deposit."
3. **Pooling:** all the pieces' numbers are averaged into **one** list of 384 numbers for the whole text.

### What is 384?

The **dimension** of the embedding: how many numbers describe a text's meaning. Think of each number as a dial measuring some aspect of meaning. Texts with similar meanings have similar dial settings, so they sit close together in "meaning space."

- 384 is fixed by the model's design (`all-MiniLM-L6-v2`). Larger models use 768, 1536 or 3072: more detail, but slower and bigger.
- **The question and the chunks must be embedded by the same model.** If you change models, update both scripts and rebuild the database.

### What the LLM does

Llama 3.2 is a "next word predictor." It reads the prompt (using its **own** tokenizer) and writes the answer one token at a time, each time picking the most likely next word. Because the prompt contains the right passages, the most likely words form a correct answer. **Good passages in, good answer out.**

> The embedding numbers are used only to **find** the right passages. The LLM never sees the numbers; it reads the original text.

### What is temperature?

A "creativity dial" for the LLM. At each step it has a list of possible next words with likelihoods (e.g. "tea" 60%, "table" 30%, "house" 10%).

| Temperature | Behaviour |
|---|---|
| **0** (this project) | Always picks the most likely word: same question, same answer, every run |
| ~0.8 (Ollama's default) | Usually the likely word, sometimes another: answers vary between runs |
| 1.5+ | Often unusual words: creative but random and error-prone |

Answering from documents needs facts, not creativity, so we use 0. Before this change, the same question once got "I don't know." and a correct answer on the next run. Note that 0 makes answers *consistent*, not automatically *correct*: a wrong answer will now be wrong every time, which makes it easier to spot and fix.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No module named 'langchain.text_splitter'` | Old import path from older tutorials | Use `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| `langchain-community is being sunset` warning | Old loader package is no longer maintained | Load files with plain Python (`Path.read_text`) as this project does |
| `'Document' object is not subscriptable` | `similarity_search` returns no scores | Use `similarity_search_with_relevance_scores` |
| `model 'llama3.2' not found (404)` | Ollama is installed but the model isn't downloaded | `ollama pull llama3.2` |
| `connection refused` | Ollama isn't running | Open the Ollama app from the Start menu |
| `Activate.ps1 cannot be loaded` | PowerShell blocks scripts | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| `HF_TOKEN` / symlinks warnings | Hugging Face notices only | Harmless; to hide the symlinks one: `$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1` |

---

## Swapping parts

Because LangChain gives every LLM the same `.invoke()` method, switching the answer-writer is a 2-line change in `query_data.py`:

```python
# Local (current)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2", temperature=0)

# Claude (needs ANTHROPIC_API_KEY, pip install langchain-anthropic)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-5-5", max_tokens=16000)
```

---

## Next steps

- **Add more documents:** drop `.md`, `.txt` or `.pdf` files in `data/books/` and rerun `create_database.py`
- **Support another type** (e.g. `.pptx`): write a loader that returns a list of `Document`s and add it to `LOADERS`, as `load_docx` does
- **Run Ragas** (`python evaluate.py all`) and record the scores here
- **Tune it:** try `chunk_size` 500–1500 and `K` 3–8
- **Try a bigger embedding model**, e.g. `all-mpnet-base-v2` (768 numbers), and rebuild the database
- **Add a web UI** with Streamlit or Gradio
