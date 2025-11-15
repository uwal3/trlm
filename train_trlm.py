import os
import time
import math
from contextlib import nullcontext
from collections import deque

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from dataset import TextDataset, collate_fn
from model.config import TRLMConfig
from model.trlm import TRLM, TRLMCarry, TRLMInnerCarry
from model.loss_head import ACTLossHead


out_dir = "out"
eval_interval = 1000
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

wandb_log = True
wandb_project = "trlm"
wandb_run_name = "trlm-" + str(time.time())

dataset_dir = "data/wikitext"
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 2
n_head = 6
n_embd = 576
dropout = 0.0
bias = False
H_cycles: int = 1
L_cycles: int = 12
halt_max_steps: int = 1
no_ACT_continue: bool = True

learning_rate = 6e-4
max_iters = 40000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False
num_workers = 4

torch.manual_seed(52)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast_mode.autocast(device_type=device_type, dtype=ptdtype)
)

os.makedirs(out_dir, exist_ok=True)

train_data = TextDataset(os.path.join(dataset_dir, "train.bin"), block_size)
val_data = TextDataset(os.path.join(dataset_dir, "val.bin"), block_size)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
)

buffer_size = batch_size * 2
train_buffer = deque(maxlen=buffer_size)
val_buffer = deque(maxlen=buffer_size)


def refill_buffer(buffer: deque, loader_iter, min_size):
    while len(buffer) < min_size:
        try:
            batch = next(loader_iter)
            l = batch["input_ids"].size(0)
            samples = [{k: v[i] for k, v in batch.items()} for i in range(l)]
            buffer.extend(samples)
        except:
            return False
    return True


model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50304,
    H_cycles=H_cycles,
    L_cycles=L_cycles,
    halt_max_steps=halt_max_steps,
    no_ACT_continue=no_ACT_continue,
)
gptconf = TRLMConfig(**model_args)
trlm: TRLM = TRLM(gptconf)
model: ACTLossHead = ACTLossHead(trlm)
model.to(device)

print(f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

if compile:
    print("compiling the model..")
    model = torch.compile(model)  # type: ignore


@torch.no_grad()
def validate():
    model.eval()
    val_iter = iter(val_loader)
    all_metrics = []

    init_batch = next(val_iter)
    init_batch = {k: v.to(device) for k, v in init_batch.items()}
    val_carry = model.initial_carry(init_batch, device)
    refill_buffer(val_buffer, val_iter, batch_size)
    new_val_samples = init_batch

    total_halted = 0
    for _ in range(eval_iters):
        val_carry, _, metrics, _, all_halted = model(
            carry=val_carry, new_samples=new_val_samples
        )
        halted_indices = torch.where(val_carry.halted)[0]
        num_halted = len(halted_indices)
        total_halted += num_halted
        all_metrics.append(metrics)

        if num_halted > 0:
            if not refill_buffer(val_buffer, val_iter, batch_size) and all_halted:
                break
            items = [val_buffer.popleft() for _ in range(num_halted)]
            new_val_samples = {
                k: torch.stack([item[k] for item in items]).to(device)
                for k in items[0].keys()
            }
        else:
            new_val_samples = None

    per_step_metrics = ["lm_loss", "q_halt_loss", "q_continue_loss"]
    final_metrics = {
        k: torch.sum(torch.stack([m[k] for m in all_metrics])).item()
        / (total_halted if k not in per_step_metrics else len(all_metrics))
        for k in all_metrics[0]
    }
    final_metrics = {f"validation/{k}": v for k, v in final_metrics.items()}

    model.train()
    return final_metrics["validation/lm_loss"], final_metrics


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log:
    config_log = {
        k: v
        for k, v in locals().items()
        if k
        in [
            "batch_size",
            "block_size",
            "n_layer",
            "H_cycles",
            "L_cycles",
            "halt_max_steps",
            "n_head",
            "n_embd",
            "learning_rate",
            "max_iters",
        ]
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=config_log)

iter_num = 0
best_val_loss = 1e9
t0 = time.time()

train_iter = iter(train_loader)
batch = next(train_iter)
batch = {k: v.to(device) for k, v in batch.items()}
refill_buffer(train_buffer, train_iter, batch_size)
carry: TRLMCarry = model.initial_carry(batch, device)
new_samples = batch

while True:
    if iter_num > max_iters:
        break

    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        lr = learning_rate

    if iter_num % eval_interval == 0:
        loss, metrics = validate()
        if wandb_log:
            wandb.log(metrics)
        if iter_num > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f"trlm-ckpt-{iter_num}.pt"))

    total_loss_acc = 0
    metrics_acc = {}

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            carry, loss, metrics, _, _ = model(carry=carry, new_samples=new_samples)

            loss = loss / gradient_accumulation_steps
            total_loss_acc += loss.detach()

        if not metrics_acc:
            metrics_acc = {k: v.detach() for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                metrics_acc[k] += v.detach()

        halted_indices = torch.where(carry.halted)[0]
        num_halted = len(halted_indices)

        if num_halted > 0:
            if not refill_buffer(train_buffer, train_iter, num_halted):
                train_iter = iter(train_loader)
                refill_buffer(train_buffer, train_iter, num_halted)

            items = [train_buffer.popleft() for _ in range(num_halted)]
            new_samples = {
                k: torch.stack([item[k] for item in items]).to(device)
                for k in items[0].keys()
            }
        else:
            new_samples = None

        detached_inner_carry = TRLMInnerCarry(
            z_H=carry.inner_carry.z_H.detach(), z_L=carry.inner_carry.z_L.detach()
        )
        detached_current_data = {k: v.detach() for k, v in carry.current_data.items()}

        carry = TRLMCarry(
            inner_carry=detached_inner_carry,
            steps=carry.steps.detach(),
            halted=carry.halted.detach(),
            current_data=detached_current_data,
        )

        loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0
    t0 = time.time()
    if iter_num % log_interval == 0:
        metrics_log = {
            f"train/{k}": v / gradient_accumulation_steps
            for k, v in metrics_acc.items()
        }
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": total_loss_acc,
                    "lr": lr,
                    **metrics_log,
                }
            )

        lossf = total_loss_acc.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

if wandb_log:
    wandb.finish()
