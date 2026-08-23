from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Load a pretrained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# Two sentences with similar meaning
sentence_1 = "Python is easy to learn"
sentence_2 = "Python is simple to learn"


# Convert the sentences into tokens and Token IDs
inputs_1 = tokenizer(sentence_1, return_tensors="pt")
inputs_2 = tokenizer(sentence_2, return_tensors="pt")


# Pass the Token IDs through BERT
# BERT produces a contextual representation for each token
with torch.no_grad():
    outputs_1 = model(**inputs_1)
    outputs_2 = model(**inputs_2)


# Get the contextual representation of each token
embeddings_1 = outputs_1.last_hidden_state
embeddings_2 = outputs_2.last_hidden_state


# BERT gives us one vector for every token.
# We use mean pooling to create one vector
# representing the whole sentence.
mask_1 = inputs_1["attention_mask"].unsqueeze(-1)
mask_2 = inputs_2["attention_mask"].unsqueeze(-1)

sentence_embedding_1 = (
    (embeddings_1 * mask_1).sum(dim=1)
    / mask_1.sum(dim=1)
)

sentence_embedding_2 = (
    (embeddings_2 * mask_2).sum(dim=1)
    / mask_2.sum(dim=1)
)


# Compare the two sentence embeddings
# using Cosine Similarity.
similarity = F.cosine_similarity(
    sentence_embedding_1,
    sentence_embedding_2
)


# Print the results
print("Sentence 1:")
print(sentence_1)

print("\nSentence 2:")
print(sentence_2)

print("\nSentence Embedding 1 shape:")
print(sentence_embedding_1.shape)

print("\nSentence Embedding 2 shape:")
print(sentence_embedding_2.shape)

print("\nCosine Similarity:")
print(similarity.item())