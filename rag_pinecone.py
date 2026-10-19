from sentence_transformers import SentenceTransformer

from pinecone_db import pinecone_retrieve

from groq_llm import generate_answer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def pinecone_rag_pipeline(question):

    print("\nSearching Pinecone...")


    results = pinecone_retrieve(
        question,
        model,
        top_k=3
    )


    if not results:
        return "No relevant information found."


    context = "\n\n".join(
        match["metadata"]["text"]
        for match in results
        if match.get("metadata")
        and match["metadata"].get("text")
    )


    if not context:
        return "No text found in retrieved documents."


    print("Generating answer using Groq...")


    answer = generate_answer(
        context,
        question
    )


    return answer


if __name__ == "__main__":

    question = "Who is Virat Kohli?"

    answer = pinecone_rag_pipeline(
        question
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)