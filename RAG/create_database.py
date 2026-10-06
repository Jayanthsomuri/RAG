import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH="data/books"
CHROMA_PATH="chroma"

# 1. Load every .md file in data/books
documents = [
    Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
    for path in Path(DATA_PATH).glob("*.md")
]

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
chunks = splitter.split_documents(documents)
print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")

# 3 + 4. Embed and save (start fresh each run)
shutil.rmtree(CHROMA_PATH, ignore_errors=True)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
print(f"Saved to {CHROMA_PATH}/")
