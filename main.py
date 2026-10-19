from rag_faiss import faiss_rag_pipeline
from rag_pinecone import pinecone_rag_pipeline


def main():

    print("=" * 60)
    print("       VIRAT KOHLI RAG APPLICATION")
    print("=" * 60)

    while True:

        print("\nChoose an option:")
        print("1. FAISS RAG")
        print("2. Pinecone RAG")
        print("3. Exit")

        choice = input("\nEnter your choice: ")


        if choice == "1":

            question = input(
                "\nEnter your question: "
            )

            answer = faiss_rag_pipeline(
                question
            )

            print("\nAnswer:")
            print(answer)


        elif choice == "2":

            question = input(
                "\nEnter your question: "
            )

            answer = pinecone_rag_pipeline(
                question
            )

            print("\nAnswer:")
            print(answer)


        elif choice == "3":

            print("Exiting...")
            break


        else:

            print("Invalid choice!")


if __name__ == "__main__":
    main()