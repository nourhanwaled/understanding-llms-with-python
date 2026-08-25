from transformers import AutoTokenizer, AutoModel
import torch


# -----------------------------------------
# 1. Load a real pretrained BERT model
# -----------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# -----------------------------------------
# 2. Input sentence
# -----------------------------------------
text = "The dog chased the cat"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)


# -----------------------------------------
# 3. Get the token embeddings
# -----------------------------------------
# These vectors represent the tokens themselves
# before adding positional information.
token_embeddings = model.embeddings.word_embeddings(
    inputs["input_ids"]
)


# -----------------------------------------
# 4. Create position IDs
# -----------------------------------------
# Each token gets a position:
# 0, 1, 2, 3, ...
position_ids = torch.arange(
    inputs["input_ids"].size(1)
).unsqueeze(0)

print("\nPosition IDs:")
print(position_ids)


# -----------------------------------------
# 5. Get BERT's learned position embeddings
# -----------------------------------------
# BERT has a learned vector for each position.
position_embeddings = model.embeddings.position_embeddings(
    position_ids
)


# -----------------------------------------
# 6. Combine token + position information
# -----------------------------------------
# The model adds the token embedding
# and its position embedding.
position_aware_embeddings = (
    token_embeddings + position_embeddings
)


# -----------------------------------------
# 7. Show the shapes
# -----------------------------------------
print("\nToken Embedding shape:")
print(token_embeddings.shape)

print("\nPosition Embedding shape:")
print(position_embeddings.shape)

print("\nPosition-aware Embedding shape:")
print(position_aware_embeddings.shape)