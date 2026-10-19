from langchain_text_splitters import RecursiveCharacterTextSplitter

from data import raw_text


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


chunks = splitter.split_text(raw_text)


if __name__ == "__main__":

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks):

        print(f"\nChunk {i}:")
        print(chunk)