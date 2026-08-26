import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# -----------------------------------------
# 1. Load a real tokenizer and pretrained BERT
# -----------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# -----------------------------------------
# 2. Input text
# -----------------------------------------
text = "The cat is sleeping."

# Convert text into Token IDs
inputs = tokenizer(text, return_tensors="pt")

print("Tokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

print("\nToken IDs:")
print(inputs["input_ids"])


# -----------------------------------------
# 3. Get the input embeddings
# -----------------------------------------
# Convert Token IDs into vectors
with torch.no_grad():
    token_embeddings = model.embeddings.word_embeddings(
        inputs["input_ids"]
    )

print("\nToken Embedding shape:")
print(token_embeddings.shape)


# -----------------------------------------
# 4. Get the positional embeddings
# -----------------------------------------
# Each position has its own learned vector
position_ids = torch.arange(
    inputs["input_ids"].size(1)
).unsqueeze(0)

with torch.no_grad():
    position_embeddings = model.embeddings.position_embeddings(
        position_ids
    )

print("\nPosition Embedding shape:")
print(position_embeddings.shape)


# -----------------------------------------
# 5. Combine token + position information
# -----------------------------------------
# The Transformer now knows:
# - what the token is
# - where the token is
with torch.no_grad():
    transformer_input = token_embeddings + position_embeddings

print("\nTransformer input shape:")
print(transformer_input.shape)


# -----------------------------------------
# 6. Pass through a Transformer layer
# -----------------------------------------
# BERT's Transformer layer contains:
# - Multi-Head Self-Attention
# - Feed Forward Network
layer = model.encoder.layer[0]

with torch.no_grad():
    layer_output = layer(
        transformer_input,
        attention_mask=None
    )[0]

print("\nTransformer layer output shape:")
print(layer_output.shape)


# -----------------------------------------
# 7. Final contextual representations
# -----------------------------------------
# Every token now has a contextual representation
# after going through the Transformer layer.
print("\nFinal contextual representation shape:")
print(layer_output.shape)