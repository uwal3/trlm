import os

import numpy as np
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


def tokenize_function(examples):
    return enc(examples["text"], add_special_tokens=False)


for split_name, ds_split in splits.items():
    print(f"processing '{split_name}' split...")

    ds_split = ds_split.filter(
        lambda x: x["text"] and not x["text"].isspace(), num_proc=8
    )

    tokenized_ds = ds_split.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )

    output_path = os.path.join(output_dir, f"{split_name}.bin")
    total_tokens = 0

    with open(output_path, "wb") as f:
        for item in tokenized_ds:
            tokens = item["input_ids"] + [enc.eos_token_id]
            f.write(np.array(tokens, dtype=np.uint16).tobytes())
            total_tokens += len(tokens)

    print(f"total tokens in '{split_name}': {total_tokens}")
    print(f"saved to {output_path}")

print("done.")
