from datasets import load_dataset

# FYI export HF_TOKEN=$(pbpaste)

# Load the shell/bash subset
dataset = load_dataset("bigcode/the-stack", split="train", data_dir="data/shell")# , lang=["bash"])
# data_dir data/shell has 11 files, not bad size (about 4GB IIEC)

# Save locally (optional)
# dataset.to_csv("shell_scripts.csv")
# print(dataset.column_names) 

subset = dataset.select(range(1))
# print(subset["content"][0])  # Example script

#
# whitespace/word tokenizer (crude):
# import re
# def simple_tokenizer(script):
#     # Split by whitespace and keep punctuation
#     tokens = re.findall(r"\w+|[^\w\s]", script, re.UNICODE)
#     return tokens
# tokenized = subset.map(lambda x: {"tokens": simple_tokenizer(x["content"])})
# print(tokenized["tokens"][0])  # Example tokenized output

from tokenizers import ByteLevelBPETokenizer

# Train the tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["path/to/shell_scripts.sh"], vocab_size=8000, min_frequency=2)

# Save and reload the tokenizer
tokenizer.save_model("tokenizer")
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")

# Tokenize the dataset
dataset = dataset.map(lambda x: {"tokens": tokenizer.encode(x["content"]).ids})
print(dataset["tokens"][0])


