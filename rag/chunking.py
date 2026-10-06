from pathlib import Path


# Load the content of a single document
def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Clean unnecessary whitespace
def clean_text(text):
    return " ".join(text.split()).strip()


# Load all text documents
def load_documents(folder_path):
    documents = []

    for file_path in Path(folder_path).glob("*.txt"):

        content = load_document(file_path)
        content = clean_text(content)

        documents.append({
            "content": content,
            "source": file_path.name
        })

    return documents


# Split text into overlapping chunks
def chunk_text(text, chunk_size=10, overlap=3):

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append({
            "content": chunk,
            "chunk_id": chunk_id
        })

        # Move forward
        start += chunk_size - overlap

        chunk_id += 1

    return chunks


# Load documents
documents = load_documents("documents")


# Create chunks
all_chunks = []

for document in documents:

    chunks = chunk_text(
        document["content"],
        chunk_size=10,
        overlap=3
    )

    print(f"\nSource: {document['source']}")
    print(f"Number of words: {len(document['content'].split())}")
    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks:

        chunk["source"] = document["source"]

        all_chunks.append(chunk)


# Display all chunks
print("\n" + "=" * 60)
print("ALL CHUNKS")
print("=" * 60)

for chunk in all_chunks:

    print(f"\nSource: {chunk['source']}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Content: {chunk['content']}")
    print("-" * 50)