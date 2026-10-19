from sentence_transformers import SentenceTransformer

from chunking import chunks


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)


if __name__ == "__main__":

    print("Embedding shape:")
    print(embeddings.shape)

    print("\nEmbedding dimension:")
    print(embeddings.shape[1])