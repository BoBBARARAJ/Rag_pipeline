import faiss

from embeddings import embeddings, model
from chunking import chunks


dimension = embeddings.shape[1]


faiss_index = faiss.IndexFlatIP(
    dimension
)


faiss_index.add(
    embeddings
)


print("FAISS index created!")
print("Total vectors:", faiss_index.ntotal)


def faiss_retrieve(query, top_k=3):

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    distances, indices = faiss_index.search(
        query_vector,
        min(top_k, len(chunks))
    )


    results = []


    for j, index in enumerate(indices[0]):

        if index == -1:
            continue

        results.append(
            {
                "text": chunks[index],
                "score": float(distances[0][j])
            }
        )


    return results