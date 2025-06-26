from transformers import AutoTokenizer

text = "Colored cats paint rats colorfully with Kei Wakabayashi."

tokenizers = {
    "WordPiece": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "BPE": AutoTokenizer.from_pretrained("gpt2"),
    "SentencePiece": AutoTokenizer.from_pretrained("xlm-roberta-base")
}

for name, tokenizer in tokenizers.items():
    print(f"\n--- {name} ---")
    tokenized = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
    print("Tokens:", tokens)
    print("Token IDs:", tokenized['input_ids'])