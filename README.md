# Local RAG: Ask Questions About a Book

A simple **Retrieval-Augmented Generation (RAG)** project that answers questions about *Alice's Adventures in Wonderland*. Everything runs **on your own computer**: no API key, no cost, and no data[...]

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

EACH TIME: Question ──► meaning-numbers ──► find 4 closest passages
                                                    │
                                                    ▼
                      Llama 3.2: "answer using only these pages" ──► Answer + Sources
```

---

## Project structure

```
.
├── .venv/                  # Python virtual environment (installed libraries; don't edit)
├── data/
│   └── books/
│       └── alice_in_wonderland.md   # source text (add more .md files here)
├── chroma/                 # vector database (created by create_database.py)
├── create_database.py      # Step 1: build the database (run once)
├── query_data.py           # Step 2: ask questions (run anytime)
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

---

## Setup (one time)

**1. Create and activate a virtual environment** (from the repo root):
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```
You should now see `(.venv)` at the start of your prompt.

**2. Install the Python libraries:**
```powershell
pip install langchain langchain-text-splitters langchain-chroma langchain-huggingface langchain-ollama sentence-transformers
```

**3. Install Ollama and download the model:**
- Download Ollama from https://ollama.com/download/windows and install it.
- Then run:
```powershell
ollama pull llama3.2
```

---

## How to run

Always run from the repo root with `(.venv)` active.

**Build the database** (run once, and again whenever you add or change books):
```powershell
python create_database.py
```
```
Split 1 document(s) into 217 chunks
Saved to chroma/
```

**Ask questions:**
```powershell
python query_data.py "How does Alice meet the Mad Hatter?"
python query_data.py "What does the Caterpillar ask Alice?"
python query_data.py "Who is Harry Potter?"     # not in the book → "No good match found" or "I don't know"
```

---

## How it works

### `create_database.py`: filing the book

| Step | Code | In plain words |
|---|---|---|
| 1. Load | `Path(DATA_PATH).glob("*.md")` | Read every `.md` file in `data/books` |
| 2. Split | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` | Cut the book into passages of about 1,000 characters. Neighbors overlap by 200 characters so no sentence is los[...] |
| 3. Embed | `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` | Turn each passage into 384 numbers that capture its meaning |
| 4. Store | `Chroma.from_documents(...)` | Save the passages and their numbers in the `chroma/` folder (the old one is deleted first) |

### `query_data.py`: answering a question

| Step | Code | In plain words |
|---|---|---|
| 1. Prompt template | `ChatPromptTemplate` | An instruction sheet: "Answer only from this context; otherwise say I don't know" |
| 2. Retrieve | `similarity_search_with_relevance_scores(question, k=4)` | Turn the question into 384 numbers and find the 4 closest passages (scored 0–1) |
| 3. Safety check | `results[0][1] < 0.3` | If even the best match is weak, stop instead of letting the AI guess |
| 4. Build context | `"\n\n---\n\n".join(...)` | Join the 4 passages into one block of text |
| 5. Generate | `ChatOllama("llama3.2").invoke(...)` | Send the filled-in prompt to Llama 3.2 and get the answer |
| 6. Show | `print(...)` | Print the answer and the source positions |

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

The **dimension** of the embedding: how many numbers describe a text's meaning. Think of each number as a dial measuring some aspect of meaning. Texts with similar meanings have similar dial sett[...]

- 384 is fixed by the model's design (`all-MiniLM-L6-v2`). Larger models use 768, 1536 or 3072: more detail, but slower and bigger.
- **The question and the chunks must be embedded by the same model.** If you change models, update both scripts and rebuild the database.

### What the LLM does

Llama 3.2 is a "next word predictor." It reads the prompt (using its **own** tokenizer) and writes the answer one token at a time, each time picking the most likely next word. Because the prompt [...]

> The embedding numbers are used only to **find** the right passages. The LLM never sees the numbers; it reads the original text.

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
llm = ChatOllama(model="llama3.2")

# Claude (needs ANTHROPIC_API_KEY, pip install langchain-anthropic)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-5-5", max_tokens=16000)
```

---

## Next steps

- **Add more books:** drop `.md` files in `data/books/` and rerun `create_database.py`
- **Remove the Project Gutenberg license text** from the book so it isn't indexed
- **Tune it:** try `chunk_size` 500–1500 and `k` 3–8
- **Try a bigger embedding model**, e.g. `all-mpnet-base-v2` (768 numbers), and rebuild the database
- **Add a web UI** with Streamlit or Gradio
