import os
import time
import math
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from dataset import TextDataset
from model.config import Config as GPTConfig
from model.gpt import GPT


out_dir = "out"
eval_interval = 1000
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

wandb_log = True
wandb_project = "trlm"
wandb_run_name = "gpt-" + str(time.time())

dataset_dir = "data/wikitext"
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 2
n_head = 6
n_embd = 576
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 25000
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
)
val_loader = DataLoader(
    val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
)

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50304,
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
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
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, (X, Y) in enumerate(val_loader):
        if k >= eval_iters:
            break
        X, Y = X.to(device), Y.to(device)
        with ctx:
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
    loss = losses.mean()
    model.train()
    return loss


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
X, Y = next(train_iter)
X, Y = X.to(device), Y.to(device)

while True:
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        lr = learning_rate

    if iter_num % eval_interval == 0:
        val_loss = estimate_loss()
        print(f"step {iter_num}: val loss {val_loss:.4f}")
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "validation/lm_loss": val_loss,
                }
            )
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f"ckpt-{iter_num}.pt"))
    total_loss = 0
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss = loss / gradient_accumulation_steps

        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
        X, Y = X.to(device), Y.to(device)

        loss.backward()
        total_loss += loss.item()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        wandb.log(
            {
                "iter": iter_num,
                "train/lm_loss": total_loss,
                "train/loss": total_loss,
                "lr": lr,
            }
        )
        print(f"iter {iter_num}: loss {total_loss:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    if iter_num > max_iters:
        break

if wandb_log:
    wandb.finish()
