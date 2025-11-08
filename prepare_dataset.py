import os
import numpy as np
import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer


dataset_name = "wikitext"
dataset_config = "wikitext-103-raw-v1"

tokenizer_name = "gpt2"

output_dir = f"data/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset(dataset_name, dataset_config, num_proc=8)

splits = {
    "train": dataset["train"],
    "val": dataset["validation"],
}

enc = AutoTokenizer.from_pretrained("gpt2")


def tokenize_and_concatenate(examples):
    full_text = enc.eos_token.join(examples["text"])

    tokens = enc.encode(full_text)

    return {"tokens": tokens}


for split_name, ds_split in splits.items():
    print(f"processing '{split_name}' split...")

    tokenized_ds = ds_split.map(
        tokenize_and_concatenate,
        batched=True,
        remove_columns=["text"],
    )

    print(f"{len(tokenized_ds)} rows tokenized")

    output_path = output_dir + f"/{split_name}.bin"

    all_tokens = np.concatenate(tokenized_ds["tokens"])

    print(f"total tokens in '{split_name}': {len(all_tokens)}")
    print(f"saving to {output_path}..")

    all_tokens.astype(np.uint16).tofile(output_path)

print("data preparation complete.")
