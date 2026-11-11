from common import load_tokenizer

tokenizer = load_tokenizer()
text = "The cats are playing."

tokens = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("Keys:", tokens.keys())
print("Tokens:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
print("Input IDs:", tokens["input_ids"][0].tolist())
print("Attention mask:", tokens["attention_mask"][0].tolist())
print("Special tokens:", tokenizer.all_special_tokens)
print("Word count:", len(text.rstrip(".").split()))
print("Content token count:", len(tokenizer.tokenize(text)))
print("Total token count:", len(tokens["input_ids"][0]))

batch = tokenizer(
    ["This movie was excellent.", "The plot was not convincing."],
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

print("Batch input IDs:", batch["input_ids"])
print("Batch attention mask:", batch["attention_mask"])
for row in batch["input_ids"]:
    print(tokenizer.convert_ids_to_tokens(row))
