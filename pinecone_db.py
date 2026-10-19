from pinecone import ServerlessSpec

from config import (
    pinecone_client,
    PINECONE_INDEX_NAME
)

from embeddings import embeddings, model
from chunking import chunks


INDEX_NAME = PINECONE_INDEX_NAME

EMBEDDING_DIMENSION = embeddings.shape[1]


def get_index():

    existing_indexes = pinecone_client.list_indexes().names()

    if INDEX_NAME not in existing_indexes:

        print("Creating Pinecone index...")

        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pinecone_client.Index(
        INDEX_NAME
    )

    return index


def upload_vectors():

    index = get_index()

    vectors = []

    for i, chunk in enumerate(chunks):

        vectors.append(
            {
                "id": str(i),

                "values": embeddings[i].tolist(),

                "metadata": {
                    "text": chunk
                }
            }
        )


    response = index.upsert(
        vectors=vectors
    )

    print("Vectors uploaded successfully!")

    return response


def pinecone_retrieve(
    query,
    model,
    top_k=3
):

    index = get_index()


    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


    results = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True
    )


    return results["matches"]