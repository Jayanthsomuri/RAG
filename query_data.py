import re
import sys
import math
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

CHROMA_PATH="chroma"
MODE="rerank"        # "vector" (meaning only), "hybrid" (+ keywords), "rerank" (+ cross-encoder)
K=5                  # passages sent to the LLM
CANDIDATES=20        # passages each search hands to the fusion/re-rank step
RRF_K=60             # standard Reciprocal Rank Fusion constant
MIN_VECTOR_SCORE=0.3 # guardrail for "vector" and "hybrid": best meaning-similarity (0-1)
MIN_RERANK_SCORE=0.5 # guardrail for "rerank": best cross-encoder score (0-1)
PROMPT = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

---

Question: {question}
If the answer is not in the context, say "I don't know."
""")

# Load once, reuse for every question
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
llm= ChatOllama(model="llama3.2", temperature=0)  # 0 = same answer every run, no random word choice
_stemmer = PorterStemmer()
_bm25 = None
_reranker = None


def tokenize(text):
    # BM25 matches words exactly, so split into lowercase words and reduce each to its stem
    # ("copyrighted" -> "copyright", "trending" -> "trend") so word forms still match
    return [_stemmer.stem(word) for word in re.findall(r"\w+", text.lower())]


def bm25_search(question, k):
    # Keyword search: build the BM25 index from every chunk stored in Chroma (once)
    global _bm25
    if _bm25 is None:
        stored = db.get(include=["documents", "metadatas"])
        docs = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(stored["documents"], stored["metadatas"])]
        _bm25 = (BM25Okapi([tokenize(d.page_content) for d in docs]), docs)
    index, docs = _bm25
    scores = index.get_scores(tokenize(question))
    best = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return [docs[i] for i in best]


def fuse(*ranked_lists):
    # Reciprocal Rank Fusion: a passage scores 1/(60 + rank) in each list it appears in.
    # Passages ranked high by both searches rise to the top. Only ranks matter, not raw scores.
    fused, by_text = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            fused[doc.page_content] = fused.get(doc.page_content, 0) + 1 / (RRF_K + rank)
            by_text[doc.page_content] = doc
    order = sorted(fused, key=lambda text: -fused[text])
    return [(by_text[text], fused[text]) for text in order]


def rerank(question, docs):
    # Cross-encoder reads question + passage together and scores relevance; squash to 0-1
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _reranker.predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: -pair[1])
    return [(doc, 1 / (1 + math.exp(-float(score)))) for doc, score in ranked]


def retrieve(question, mode=MODE):
    # Returns (top K (Document, score) pairs, whether the guardrail lets them through)
    vector = db.similarity_search_with_relevance_scores(question, k=CANDIDATES)
    best_vector = vector[0][1] if vector else 0
    if mode == "vector":
        return vector[:K], best_vector >= MIN_VECTOR_SCORE

    hybrid = fuse([doc for doc, _ in vector], bm25_search(question, CANDIDATES))
    if mode == "hybrid":
        return hybrid[:K], best_vector >= MIN_VECTOR_SCORE

    reranked = rerank(question, [doc for doc, _ in hybrid])
    return reranked[:K], bool(reranked) and reranked[0][1] >= MIN_RERANK_SCORE


def ask(question, mode=MODE):
    # Returns (answer text, list of (Document, score)). Answer is "No good match found" if retrieval is weak.
    results, confident = retrieve(question, mode)
    if not confident:
        return "No good match found", results

    context= "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    #generate answer
    answer = llm.invoke(PROMPT.format(context=context, question=question))
    return answer.content, results


if __name__ == "__main__":
    answer, results = ask(sys.argv[1])
    print(answer)

    # Show which file (and page, for PDFs) each passage came from
    print("\nSources:")
    for doc, score in results:
        name = Path(doc.metadata.get("source", "?")).name
        page = f", page {doc.metadata['page']}" if "page" in doc.metadata else ""
        print(f"  - {name}{page} (start {doc.metadata.get('start_index')}, score {score:.2f})")
