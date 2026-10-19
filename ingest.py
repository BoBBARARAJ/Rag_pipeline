from pinecone_db import upload_vectors


if __name__ == "__main__":

    print("Starting Pinecone ingestion...")

    upload_vectors()

    print("Ingestion completed!")