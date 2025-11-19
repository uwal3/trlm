from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str] = [],
        **model_kwargs,
    ) -> Tuple[
        Any,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
        torch.Tensor,
    ]:
        new_carry, outputs = self.model(**model_kwargs)
        target = new_carry.current_data["target"]

        with torch.no_grad():
            sequence_length = target.size(1)
            preds = torch.argmax(outputs["logits"], dim=-1)

            is_correct = preds == target
            seq_is_correct = is_correct.sum(-1) == sequence_length

        logits = outputs["logits"].to(torch.float32)
        q_halt_logits = outputs["q_halt_logits"].to(torch.float32)

        # losses
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
        )
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits,
            seq_is_correct.to(q_halt_logits.dtype),
            reduction="sum",
        )
        metrics = {
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        }

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
