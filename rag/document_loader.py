import re
from pathlib import Path


# Load the content of a single document
def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Clean unnecessary whitespace from the text
def clean_text(text):
    # Replace multiple spaces and tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple new lines with a single new line
    text = re.sub(r"\n+", "\n", text)

    # Remove whitespace from the beginning and end
    return text.strip()


# Load all text documents from a folder
def load_documents(folder_path):
    documents = []

    # Find all .txt files inside the folder
    for file_path in Path(folder_path).glob("*.txt"):

        # Load the document content
        content = load_document(file_path)

        # Clean the extracted text
        content = clean_text(content)

        # Store the content and its source
        documents.append({
            "content": content,
            "source": file_path.name
        })

    return documents


# Load and clean all documents
documents = load_documents("documents")


# Display the loaded documents
for document in documents:
    print(f"Source: {document['source']}")
    print(document["content"])
    print("-" * 40)