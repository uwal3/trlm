import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):
    def __init__(self, data_file, block_size):
        super().__init__()
        self.block_size = block_size

        self.data = np.memmap(data_file, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]

        full_sequence = torch.from_numpy(chunk.astype(np.int64))

        x = full_sequence[:-1]
        y = full_sequence[1:]

        return x, y
