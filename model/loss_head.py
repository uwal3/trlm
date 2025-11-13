from typing import Any, Tuple, Dict, Sequence, Optional

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
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        target = new_carry.current_data["target"]

        with torch.no_grad():
            sequence_length = target.size(1)
            preds = torch.argmax(outputs["logits"], dim=-1)

            is_correct = preds == target
            seq_is_correct = is_correct.sum(-1) == sequence_length

            # metrics (halted)
            valid_metrics = new_carry.halted
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, is_correct.float().mean(-1), 0
                ).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        logits = outputs["logits"].to(torch.float32)
        q_halt_logits = outputs["q_halt_logits"].to(torch.float32)
        q_continue_logits = outputs["q_continue_logits"].to(torch.float32)

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
        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits,
                outputs["target_q_continue"],
                reduction="sum",
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
