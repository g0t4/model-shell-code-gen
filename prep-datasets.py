from datasets import load_dataset

# FYI export HF_TOKEN=$(pbpaste)

# Load the shell/bash subset
dataset = load_dataset("bigcode/the-stack", split="train", data_dir="data/shell")# , lang=["bash"])
# data_dir data/shell has 11 files, not bad size (about 4GB IIEC)

# Save locally (optional)
# dataset.to_csv("shell_scripts.csv")
# print(dataset.column_names) 

import re

def simple_tokenizer(script):
    # Split by whitespace and keep punctuation
    tokens = re.findall(r"\w+|[^\w\s]", script, re.UNICODE)
    return tokens

subset = dataset.select(range(100))
print(subset["content"][0])  # Example script
# Apply tokenizer to the dataset
# dataset = dataset.map(lambda x: {"tokens": simple_tokenizer(x["content"])})
# print(dataset["tokens"][0])  # Example tokenized output


