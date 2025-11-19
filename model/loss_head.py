from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LossHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch, **model_kwargs) -> torch.Tensor:
        logits = self.model(batch["input_ids"], **model_kwargs).to(torch.float32)
        target = batch["target"]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
        )

        return loss
