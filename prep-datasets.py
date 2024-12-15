from datasets import load_dataset

# FYI export HF_TOKEN=$(pbpaste)

# Load the shell/bash subset
dataset = load_dataset("bigcode/the-stack", split="train", languages=["bash"])

# Save locally (optional)
dataset.to_csv("shell_scripts.csv")
