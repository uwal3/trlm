import os

import hydra
import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer

from datasets import load_dataset


@hydra.main(version_base=None, config_path="../conf/data", config_name="the_pile.yaml")
def main(cfg: DictConfig):
    cfg = cfg._group_

    output_dir = cfg.dir
    os.makedirs(output_dir, exist_ok=True)

    RESERVE_EVERY_N_DOCS = int(1 / cfg.validation_ratio)

    dataset = load_dataset(cfg.name, cfg.config, streaming=True, trust_remote_code=True)

    enc = AutoTokenizer.from_pretrained(cfg.tokenizer)

    def tokenize_function(examples):
        text = examples.get("text")
        if text:
            return enc(text, add_special_tokens=False)
        return {"input_ids": []}

    ds_split = dataset["train"]  # type: ignore

    ds_split = ds_split.filter(lambda x: x.get("text") and not x["text"].isspace())  # type: ignore

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
        for i in range(cfg.num_documents):
            try:
                item = next(iterator)
            except StopIteration:
                print("premature end of data :/")
                break

            if not (item and item.get("input_ids")):  # type: ignore
                continue

            tokens = item["input_ids"] + [enc.eos_token_id]  # type: ignore
            tokens_bytes = np.array(tokens, dtype=np.uint16).tobytes()

            if (i + 1) % RESERVE_EVERY_N_DOCS == 0:
                f_val.write(tokens_bytes)
                val_tokens += len(tokens)
            else:
                f_train.write(tokens_bytes)
                train_tokens += len(tokens)

            if (i + 1) % 10000 == 0:
                print(f"processed {i+1}/{cfg.num_documents} documents...")

    print(f"total tokens in train: {train_tokens}")
    print(f"total tokens in validation: {val_tokens}")
    print("done.")


if __name__ == "__main__":
    main()
