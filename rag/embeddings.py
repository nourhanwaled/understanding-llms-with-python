import chromadb
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Create or get a collection
collection = client.get_or_create_collection(
    name="documents"
)


# Our chunks
chunks = [
    {
        "content": "Our refund policy allows customers to request a refund within 30 days of purchase.",
        "source": "refund_policy.txt",
        "chunk_id": 0
    },
    {
        "content": "Refund requests must be submitted through the customer support portal.",
        "source": "refund_policy.txt",
        "chunk_id": 1
    },
    {
        "content": "Refunds are processed within 5 to 7 business days.",
        "source": "refund_policy.txt",
        "chunk_id": 2
    },
    {
        "content": "Standard shipping usually takes 3 to 5 business days.",
        "source": "shipping_policy.txt",
        "chunk_id": 0
    },
    {
        "content": "Express shipping is available and usually takes 1 to 2 business days.",
        "source": "shipping_policy.txt",
        "chunk_id": 1
    },
    {
        "content": "We respect the privacy of our customers and protect their personal information.",
        "source": "privacy_policy.txt",
        "chunk_id": 0
    }
]


# Extract text from chunks
texts = [
    chunk["content"]
    for chunk in chunks
]


# Generate embeddings
embeddings = model.encode(texts).tolist()


# Store chunks in ChromaDB
collection.upsert(
    ids=[
        f"{chunk['source']}_{chunk['chunk_id']}"
        for chunk in chunks
    ],
    embeddings=embeddings,
    documents=texts,
    metadatas=[
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]
)


print("Documents stored in ChromaDB.")


# User question
question = "How long does a refund take?"


# Generate embedding for the question
question_embedding = model.encode(
    [question]
).tolist()


# Search for similar chunks
results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)


# Display results
print("\nSimilarity Search Results:")

for i in range(len(results["documents"][0])):

    print(f"\nResult {i + 1}")

    print(
        f"Distance: "
        f"{results['distances'][0][i]:.4f}"
    )

    print(
        f"Source: "
        f"{results['metadatas'][0][i]['source']}"
    )

    print(
        f"Chunk ID: "
        f"{results['metadatas'][0][i]['chunk_id']}"
    )

    print(
        f"Content: "
        f"{results['documents'][0][i]}"
    )