from faiss_db import faiss_retrieve

from groq_llm import generate_answer


def faiss_rag_pipeline(question):

    print("\nSearching FAISS...")

    results = faiss_retrieve(
        question,
        top_k=3
    )


    if not results:
        return "No relevant information found."


    context = "\n\n".join(
        result["text"]
        for result in results
        if result.get("text")
    )


    if not context:
        return "No text found in retrieved documents."


    print("Generating answer using Groq...")


    answer = generate_answer(
        context,
        question
    )


    return answer