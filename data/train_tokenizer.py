from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["tokenizer_corpus.txt"],
    vocab_size=8192,       # Small vocab for small model
    min_frequency=2,
    special_tokens=["<pad>", "<eos>", "<bos>", "<problem>", "</problem>", "<reasoning>", "</reasoning>", "<answer>", "</answer>"]
)

tokenizer.save("gsm8k_tokenizer.json")
print("Tokenizer saved to gsm8k_tokenizer.json")
