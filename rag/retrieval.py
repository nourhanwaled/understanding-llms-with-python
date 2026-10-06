import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Load the embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Connect to the existing ChromaDB
# --------------------------------------------------

# ChromaDB will use the database created
# in embeddings.py
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Get the existing collection
collection = client.get_collection(
    name="documents"
)


# --------------------------------------------------
# User question
# --------------------------------------------------

question = "How long does a refund take?"


# Generate an embedding for the question
question_embedding = model.encode(
    [question]
)[0]


# --------------------------------------------------
# 1. Top-K Retrieval
# --------------------------------------------------

# Retrieve the top 3 most similar chunks
#
# n_results=3 means:
# "Return the 3 most relevant chunks"
#
# This is what we call Top-K Retrieval.
results = collection.query(
    query_embeddings=[
        question_embedding.tolist()
    ],
    n_results=3
)


print("\nTop-K Retrieval Results:")


# Display the retrieved chunks
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


# --------------------------------------------------
# 2. Cosine Similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """
    Calculate how similar two vectors are.

    A higher value means the vectors are
    more similar in meaning.
    """

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )


# --------------------------------------------------
# 3. MMR - Maximal Marginal Relevance
# --------------------------------------------------

def mmr(
    query_embedding,
    document_embeddings,
    top_k=3,
    lambda_value=0.5
):
    """
    Select relevant and diverse documents.

    lambda_value controls the balance between
    relevance and diversity.

    lambda = 1.0
        Focus more on relevance.

    lambda = 0.0
        Focus more on diversity.

    lambda = 0.5
        Balance between relevance and diversity.
    """

    selected = []

    # Continue selecting documents until
    # we reach the requested number of results.
    while len(selected) < min(
        top_k,
        len(document_embeddings)
    ):

        best_score = -float("inf")
        best_index = None

        # Check every document that has not
        # already been selected.
        for i, document_embedding in enumerate(
            document_embeddings
        ):

            # Skip documents that were already selected
            if i in selected:
                continue

            # ------------------------------------------
            # Relevance
            # ------------------------------------------

            # How similar is this document
            # to the user's question?
            relevance = cosine_similarity(
                query_embedding,
                document_embedding
            )

            # ------------------------------------------
            # Diversity
            # ------------------------------------------

            if not selected:

                # For the first document there are
                # no selected documents to compare with.
                diversity = 0

            else:

                # Find the highest similarity between
                # this document and the documents
                # already selected.
                diversity = max(
                    cosine_similarity(
                        document_embedding,
                        document_embeddings[j]
                    )
                    for j in selected
                )

            # ------------------------------------------
            # MMR Score
            # ------------------------------------------

            # MMR balances:
            #
            # Relevance → similarity to the question
            #
            # Diversity → avoid selecting documents
            # that are too similar to selected ones.
            score = (
                lambda_value * relevance
                - (1 - lambda_value) * diversity
            )

            # Keep the document with the
            # highest MMR score.
            if score > best_score:

                best_score = score
                best_index = i

        # Add the best document to the selected list
        selected.append(best_index)

    return selected


# --------------------------------------------------
# 4. Get all stored embeddings
# --------------------------------------------------

# Get all documents, metadata and embeddings
# stored in our ChromaDB collection.
all_data = collection.get(
    include=[
        "documents",
        "metadatas",
        "embeddings"
    ]
)


# Convert embeddings to a NumPy array
document_embeddings = np.array(
    all_data["embeddings"]
)


# --------------------------------------------------
# 5. Apply MMR
# --------------------------------------------------

# Select the top 3 documents using MMR.
#
# lambda_value=0.5 means:
# 50% focus on relevance
# 50% focus on diversity
selected_indexes = mmr(
    query_embedding=question_embedding,
    document_embeddings=document_embeddings,
    top_k=3,
    lambda_value=0.5
)


# --------------------------------------------------
# 6. Display MMR Results
# --------------------------------------------------

print("\nMMR Results:")


for i, index in enumerate(selected_indexes):

    print(f"\nResult {i + 1}")

    print(
        f"Source: "
        f"{all_data['metadatas'][index]['source']}"
    )

    print(
        f"Chunk ID: "
        f"{all_data['metadatas'][index]['chunk_id']}"
    )

    print(
        f"Content: "
        f"{all_data['documents'][index]}"
    )