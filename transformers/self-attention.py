from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# -----------------------------------------
# 1. Load a real pretrained BERT model
# -----------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# -----------------------------------------
# 2. Input sentence
# -----------------------------------------
text = "The animal didn't cross the road because it was tired."

# Convert the sentence into Token IDs
inputs = tokenizer(text, return_tensors="pt")

# Get the actual tokens so we can read
# the attention results later
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)


# -----------------------------------------
# 3. Get the input representations
# -----------------------------------------
# The Token IDs are converted into
# input vectors that enter the Transformer layer.
with torch.no_grad():
    embedding_output = model.embeddings(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs["token_type_ids"]
    )


# -----------------------------------------
# 4. Get one Self-Attention layer
# -----------------------------------------
# We use one Transformer layer only
# to keep the example simple.
attention = model.encoder.layer[0].attention.self


# -----------------------------------------
# 5. Create Query, Key and Value
# -----------------------------------------
# The same token representations are transformed
# into three different representations:
#
# Query → What am I looking for?
# Key   → What information do I represent?
# Value → What information can I provide?
with torch.no_grad():
    Q = attention.query(embedding_output)
    K = attention.key(embedding_output)
    V = attention.value(embedding_output)


# -----------------------------------------
# 6. Focus on the token "it"
# -----------------------------------------
# We only want to calculate the attention
# FROM the token "it".
target_index = tokens.index("it")


# IMPORTANT:
# Q has shape:
# [batch_size, number_of_tokens, vector_size]
#
# Q[0, target_index] means:
# - 0 → first batch (we have only one sentence)
# - target_index → the position of "it"
#
# So here we are selecting ONLY the Query
# belonging to "it".
query_it = Q[0, target_index]


# -----------------------------------------
# 7. Keep the Keys and Values for ALL tokens
# -----------------------------------------
# K has shape:
# [batch_size, number_of_tokens, vector_size]
#
# K[0] means:
# - take the first batch
# - KEEP ALL tokens
#
# It does NOT mean "take the first token".
#
# After K[0], we have one Key vector
# for EVERY token in the sentence.
keys = K[0]


# V has the same idea:
# V[0] means:
# - take the first batch
# - KEEP ALL tokens
#
# So we have one Value vector
# for EVERY token in the sentence.
values = V[0]


# -----------------------------------------
# 8. Compare "it" with EVERY token
# -----------------------------------------
# The Query of "it" is compared with
# the Key of every token in the sentence.
#
# This tells us how relevant each token
# is to "it".
scores = torch.matmul(keys, query_it)

# Scale the scores before Softmax
scores = scores / (keys.shape[-1] ** 0.5)


# -----------------------------------------
# 9. Convert scores into Attention Weights
# -----------------------------------------
# Softmax converts the scores into weights
# between 0 and 1.
#
# The weights show how much attention
# "it" gives to each token.
attention_weights = F.softmax(scores, dim=0)


# -----------------------------------------
# 10. Print the Attention Weights
# -----------------------------------------
print("\nAttention Weights for 'it':")

for token, weight in zip(tokens, attention_weights):
    print(f"{token:15} {weight.item():.4f}")


# -----------------------------------------
# 11. Build the Contextual Representation
# -----------------------------------------
# Each token's Value is multiplied by
# the attention weight it received.
#
# Then we add all of these weighted Values.
#
# The result is a NEW VECTOR for "it"
# that contains information from its context.
contextual_representation = torch.sum(
    attention_weights.unsqueeze(-1) * values,
    dim=0
)


# -----------------------------------------
# 12. Show the new representation
# -----------------------------------------
print("\nContextual Representation shape:")
print(contextual_representation.shape)

print("\nContextual Representation for 'it':")
print(contextual_representation)