from fastapi import FastAPI
from pydantic import BaseModel
from RAG_vLLM import RAGpipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

rag = RAGpipeline(doc_path="docs/sample.txt")

@app.post("/rag")
def rag_query(request: QueryRequest):
    answer = rag.generate(request.query)
    return {"response": answer}


