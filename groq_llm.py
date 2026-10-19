from config import groq_client


MODEL_NAME = "openai/gpt-oss-20b"


def generate_answer(context, question):

    prompt = f"""
You are a helpful assistant for a Virat Kohli RAG application.

Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{question}

Rules:
- Use only the context.
- Do not use outside knowledge.
- Do not make up facts.
- Keep the answer clear and concise.
- If the answer is not available in the context, say:
"I don't know based on the provided information."

Answer:
"""


    response = groq_client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )


    return response.choices[0].message.content