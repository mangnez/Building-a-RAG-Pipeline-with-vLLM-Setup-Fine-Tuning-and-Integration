'''
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="models"
)

--------------------------------------------------------------------

import requests

response = requests.post(
    "http://localhost:8080/v1/completions",
    headers={"Content-Type": "application/json"},
    json={
        "prompt": "What is gravity?",
        "max_tokens": 100
    }
)

print(response.json()["choices"][0]["text"])
'''

import os
import re
from collections import deque
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Set your local vLLM server endpoint
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
os.environ["OPENAI_API_KEY"] = "not-needed"

class RAGpipeline:
    def __init__(self, doc_path, embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.doc_path = doc_path
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.history = deque(maxlen=100)
        self.vectorstore = None

    def clean(self, text):
        text = re.sub(r"\n+", " ", text)
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def chunk_and_embed(self):
        if self.doc_path.endswith(".txt"):
            loader = TextLoader(self.doc_path, encoding="utf-8")
        else:
            loader = PyPDFLoader(self.doc_path)
        docs = loader.load()
        text = " ".join([doc.page_content for doc in docs])
        sentences = self.clean(text)
        chunks = [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
        embeddings = self.encoder.encode(chunks)
        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vectorstore = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name=self.embedding_model))
        return chunks

    def retrieve(self, query, top_k=5):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        return retriever.get_relevant_documents(query)

    def generate(self, query, top_k=5):
        docs = self.retrieve(query, top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        history_text = "\n\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in self.history])

        prompt = f"""
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Previous conversation:
{history_text}

Context:
{context}

Question:
{query}
"""
        llm = OpenAI(model_name="mistral", temperature=0.3)
        response = llm.invoke(prompt)
        self.history.append({"query": query, "response": response})
        return response

if __name__ == "__main__":
    doc_path = "XYZ_doc.txt"
    rag = RAGpipeline(doc_path)
    rag.chunk_and_embed()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag.generate(query)
        print("\nAnswer:\n", answer)
