import os
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

# build corpus
if not os.path.exists("tmp"):
    os.makedirs("tmp")

corpus_file = "tmp/shell_scripts_corpus.sh"
if not os.path.exists(corpus_file):
    with open(corpus_file, "w") as f:
        for example in subset["content"]:  # Adjust "content" to match your dataset key
            f.write(example + "\n")

from tokenizers import ByteLevelBPETokenizer

tokenizer_path = "tmp/trained-tokenizer"
if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path)

# Train the tokenizer
if not os.path.exists(tokenizer_path + "/vocab.json"):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[corpus_file], vocab_size=8000, min_frequency=2) # PRN adjust vocab_size/min_frequency? 
    tokenizer.save_model(tokenizer_path)

# load the tokenizer
tokenizer = ByteLevelBPETokenizer(tokenizer_path + "/vocab.json", tokenizer_path + "/merges.txt")

# Tokenize the dataset
subset_tokenizd = subset.map(lambda x: {"tokens": tokenizer.encode(x["content"]).ids})
print(subset_tokenizd["tokens"][0])  # Example tokenized output
print()
#
# # VIEW SOME TOKENS:
# # show each token for first 10:
# for i in range(100):
#     print(subset_tokenizd["tokens"][0][i], tokenizer.decode([subset_tokenizd["tokens"][0][i]]))



def create_training_pairs(tokens, seq_len):
    # Split into sequences of length seq_len + 1
    sequences = [tokens[i:i + seq_len + 1] for i in range(len(tokens) - seq_len)]
    return sequences

seq_len = 50
pairs = subset_tokenizd.map(lambda x: {"sequences": create_training_pairs(x["tokens"], seq_len)})
for i in pairs["sequences"][0]:
    print(tokenizer.decode(i))




