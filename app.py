from fastapi import FastAPI
from pydantic import BaseModel

from rag_pinecone import pinecone_rag_pipeline


app = FastAPI(
    title="Virat Kohli RAG API",
    description="RAG API using Pinecone and Groq",
    version="1.0"
)


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def home():

    return {
        "message": "Virat Kohli RAG API is running"
    }


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    answer = pinecone_rag_pipeline(
        request.question
    )


    return {
        "question": request.question,
        "answer": answer
    }