import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from datasets import load_dataset


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    
    output_dir = cfg.data.dir
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(cfg.data.name, cfg.data.config, num_proc=cfg.data.num_workers)

    splits = {
        "train": dataset["train"],  # type: ignore
        "val": dataset["validation"],  # type: ignore
    }

    enc = AutoTokenizer.from_pretrained(cfg.data.tokenizer)

    def tokenize_function(examples):
        return enc(examples["text"], add_special_tokens=False)

    for split_name, ds_split in splits.items():
        print(f"processing '{split_name}' split...")

        ds_split = ds_split.filter(  # type: ignore
            lambda x: x["text"] and not x["text"].isspace(), num_proc=cfg.data.num_workers
        )

        tokenized_ds = ds_split.map(
            tokenize_function,
            batched=True,
            num_proc=cfg.data.num_workers,
            remove_columns=["text"],
        )

        output_path = os.path.join(output_dir, f"{split_name}.bin")
        total_tokens = 0

        with open(output_path, "wb") as f:
            for item in tokenized_ds:
                tokens = item["input_ids"] + [enc.eos_token_id]  # type: ignore
                f.write(np.array(tokens, dtype=np.uint16).tobytes())
                total_tokens += len(tokens)

        print(f"total tokens in '{split_name}': {total_tokens}")
        print(f"saved to {output_path}")

    print("done.")


if __name__ == "__main__":
    main()
