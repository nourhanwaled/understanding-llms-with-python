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

# Get the actual tokens
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)


# -----------------------------------------
# 3. Get the Multi-Head Attention module
#    from the first Transformer layer
# -----------------------------------------
# BERT's Self-Attention module contains
# multiple attention heads.
# We use one Transformer layer to keep
# the example simple.
layer = model.encoder.layer[0]

attention = layer.attention.self


# -----------------------------------------
# 4. Get the input representations
# -----------------------------------------
# Token IDs are converted into the vectors
# that enter the Transformer layer.
with torch.no_grad():
    embedding_output = model.embeddings(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs["token_type_ids"]
    )


# -----------------------------------------
# 5. Create Query, Key and Value
# -----------------------------------------
# The same input representations are transformed
# into separate Query, Key and Value representations.
with torch.no_grad():
    Q = attention.query(embedding_output)
    K = attention.key(embedding_output)
    V = attention.value(embedding_output)


# -----------------------------------------
# 6. Split the vectors into multiple heads
# -----------------------------------------
# BERT-base uses 12 attention heads.
#
# Instead of one large attention operation,
# the model splits the representation into
# smaller parts and processes them in parallel.
batch_size, seq_length, hidden_size = Q.shape

num_heads = attention.num_attention_heads
head_size = hidden_size // num_heads

print("\nNumber of attention heads:")
print(num_heads)

print("\nHidden size:")
print(hidden_size)

print("\nSize of each head:")
print(head_size)


# Reshape so each head has its own Q, K and V
Q = Q.view(
    batch_size,
    seq_length,
    num_heads,
    head_size
).transpose(1, 2)

K = K.view(
    batch_size,
    seq_length,
    num_heads,
    head_size
).transpose(1, 2)

V = V.view(
    batch_size,
    seq_length,
    num_heads,
    head_size
).transpose(1, 2)


# -----------------------------------------
# 7. Calculate Attention Scores
# -----------------------------------------
# Every head compares its Query with
# the Keys of all tokens.
scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)

# Scale the scores before Softmax
scores = scores / (head_size ** 0.5)


# -----------------------------------------
# 8. Convert scores into Attention Weights
# -----------------------------------------
# Softmax turns the scores into weights
# for every token in every attention head.
attention_weights = F.softmax(
    scores,
    dim=-1
)


# -----------------------------------------
# 9. Use the Attention Weights with Values
# -----------------------------------------
# Each head creates its own contextual
# representation using its own attention weights.
head_outputs = torch.matmul(
    attention_weights,
    V
)


# -----------------------------------------
# 10. Combine all attention heads
# -----------------------------------------
# Each head learned a different representation
# of the same input.
#
# We combine the outputs of all heads to create
# one richer representation for each token.
combined_output = head_outputs.transpose(
    1, 2
).contiguous().view(
    batch_size,
    seq_length,
    hidden_size
)


# -----------------------------------------
# 11. Show the result
# -----------------------------------------
print("\nHead output shape:")
print(head_outputs.shape)

print("\nCombined output shape:")
print(combined_output.shape)