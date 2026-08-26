import tiktoken
from transformers import AutoTokenizer

text = "unbelievable"


# -----------------------------
# 1. OpenAI's tiktoken
# -----------------------------

encoding = tiktoken.get_encoding("cl100k_base")

token_ids = encoding.encode(text)

print("Tiktoken")
print("Tokens IDs:", token_ids)
print("Number of tokens:", len(token_ids))

tokens = [
    encoding.decode_single_token_bytes(token_id).decode(
        "utf-8",
        errors="replace"
    )
    for token_id in token_ids
]

print("Tokens:", tokens)


# -----------------------------
# 2. BERT Tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nBERT")
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))