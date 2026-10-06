import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

CHROMA_PATH="chroma"
PROMPT = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

---

Question: {question}
If the answer is not in the context, say "I don't know."
""")

question = sys.argv[1]

# Retrieve
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
results=db.similarity_search_with_relevance_scores(question, k=4)
if not results or results[0][1]<0.3:
    print("No good match found")
    sys.exit()

context= "\n\n---\n\n".join(doc.page_content for doc, _ in results)

#generate answer
llm= ChatOllama(model="llama3.2")
answer = llm.invoke(PROMPT.format(context=context, question=question))
print(answer.content)
print("\nSources:", [doc.metadata.get("start_index") for doc, _ in results])

