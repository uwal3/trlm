import os
import numpy as np
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
from itertools import islice

dataset_name = "EleutherAI/the_pile_deduplicated"
dataset_config = "all"

tokenizer_name = "EleutherAI/gpt-neox-20b"

output_dir = "data/the_pile"
os.makedirs(output_dir, exist_ok=True)

NUM_DOCUMENTS_TO_PROCESS = 300_000
VAL_RATIO = 0.005  # 0.5%
RESERVE_EVERY_N_DOCS = int(1 / VAL_RATIO)

dataset = load_dataset(
    dataset_name, dataset_config, streaming=True, trust_remote_code=True
)

enc = AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_function(examples):
    text = examples.get("text")
    if text:
        return enc(text, add_special_tokens=False)
    return {"input_ids": []}


ds_split = dataset["train"]

ds_split = ds_split.filter(lambda x: x.get("text") and not x["text"].isspace())

tokenized_ds = ds_split.map(
    tokenize_function,
    batched=False,
    remove_columns=["text", "meta"],
)

train_output_path = os.path.join(output_dir, "train.bin")
val_output_path = os.path.join(output_dir, "val.bin")

train_tokens = 0
val_tokens = 0

iterator = iter(tokenized_ds)

with open(train_output_path, "wb") as f_train, open(val_output_path, "wb") as f_val:

    iterator = iter(tokenized_ds)
    for i in range(NUM_DOCUMENTS_TO_PROCESS):
        try:
            item = next(iterator)
        except StopIteration:
            print("Датасет закончился раньше, чем ожидалось.")
            break

        if not (item and item.get("input_ids")):
            continue

        tokens = item["input_ids"] + [enc.eos_token_id]
        tokens_bytes = np.array(tokens, dtype=np.uint16).tobytes()

        if (i + 1) % RESERVE_EVERY_N_DOCS == 0:
            f_val.write(tokens_bytes)
            val_tokens += len(tokens)
        else:
            f_train.write(tokens_bytes)
            train_tokens += len(tokens)

        if (i + 1) % 10000 == 0:
            print(f"processed {i+1}/{NUM_DOCUMENTS_TO_PROCESS} documents...")

print(f"\ntotal tokens in train: {train_tokens}")
print(f"\ntotal tokens in validation: {val_tokens}")
print(f"saved to {train_output_path}, {val_output_path}")
print("done.")
