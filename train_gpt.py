import math
import os
import time
from contextlib import nullcontext

import hydra
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import TextDataset
from model.config import Config as GPTConfig
from model.gpt import GPT


def setup_environment(cfg: DictConfig):
    device = cfg.environment.device
    dtype = cfg.environment.dtype

    torch.manual_seed(52)
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
    return ctx


def prepare_dataloaders(cfg: DictConfig):
    dataset_dir = os.path.join(get_original_cwd(), cfg.data.dir)
    block_size = cfg.data.block_size

    train_data = TextDataset(os.path.join(dataset_dir, "train.bin"), block_size)
    val_data = TextDataset(os.path.join(dataset_dir, "val.bin"), block_size)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)

    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.data.block_size,
        bias=cfg.model.bias,
        vocab_size=tokenizer.vocab_size,
    )
    gptconf = GPTConfig(**model_args)  # type: ignore
    model = GPT(gptconf)
    model.to(cfg.environment.device)

    return model


def get_lr(iter_num, cfg):
    lr_decay_iters = cfg.optimizer.lr_decay_iters
    warmup_iter = cfg.optimizer.warmup_iters
    min_lr = cfg.optimizer.min_lr

    if iter_num < warmup_iter:
        return cfg.optimizer.learning_rate * iter_num / warmup_iter
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iter) / (lr_decay_iters - warmup_iter)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (cfg.optimizer.learning_rate - min_lr)


@torch.no_grad()
def val(model, val_loader, cfg, ctx):
    eval_iters = cfg.training.eval_iters * cfg.data.gradient_accumulation_steps
    total_loss = 0

    val_iter = iter(val_loader)
    X, Y = next(val_iter)
    X, Y = X.to(cfg.environment.device), Y.to(cfg.environment.device)

    model.eval()
    for _ in range(eval_iters):
        with ctx:
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

            total_loss += loss.item()

        try:
            X, Y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            X, Y = next(val_iter)
        X, Y = X.to(cfg.environment.device), Y.to(cfg.environment.device)

    return total_loss / eval_iters


def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    cfg,
    ctx=nullcontext(),
    iter_num=0,
    best_val_loss=1e9,
):

    train_iter = iter(train_loader)
    X, Y = next(train_iter)
    X, Y = X.to(cfg.environment.device), Y.to(cfg.environment.device)

    while True:
        t0 = time.time()

        if iter_num >= cfg.training.max_iters:
            break
        if cfg.optimizer.decay_lr:
            lr = get_lr(iter_num, cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if iter_num % cfg.training.eval_interval == 0:
            val_loss = val(model=model, val_loader=val_loader, cfg=cfg, ctx=ctx)
            if val_loss < best_val_loss or cfg.training.always_save_checkpoint:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": OmegaConf.to_container(cfg.model, resolve=True),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "wandb_run_id": wandb.run.id if cfg.wandb.log else None,
                }
                print(f"saving checkpoint to {cfg.training.out_dir}")
                torch.save(
                    checkpoint,
                    os.path.join(
                        os.path.join(get_original_cwd(), cfg.training.out_dir),
                        f"ckpt-{iter_num}.pt",
                    ),
                )
            print(f"validation loss: {val_loss:2f}")
            if cfg.wandb.log:
                wandb.log({"validation/loss": val_loss}, step=iter_num)

        total_loss = 0
        for micro_step in range(cfg.data.gradient_accumulation_steps):
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                loss = loss / cfg.data.gradient_accumulation_steps

                total_loss += loss.item()

            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
            X, Y = X.to(cfg.environment.device), Y.to(cfg.environment.device)

            loss.backward()

        if cfg.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg.training.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {iter_num}: loss {total_loss:.4f}, time {dt*1000:.2f}ms")
            if cfg.wandb.log:
                wandb.log(
                    {
                        "train/loss": total_loss,
                        "train/lm_loss": total_loss,
                        "lr": lr,
                    },
                    step=iter_num,
                )
        iter_num += 1


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):

    checkpoint = None
    wandb_run_id = None
    if cfg.training.resume_from_checkpoint:
        print(
            f"ignoring current config, loading cfg used in {cfg.training.resume_from_checkpoint}"
        )
        checkpoint_path = os.path.join(
            get_original_cwd(), cfg.training.resume_from_checkpoint
        )
        checkpoint = torch.load(checkpoint_path)
        cfg = OmegaConf.create(checkpoint["cfg"])  # type: ignore
        wandb_run_id = checkpoint.get("wandb_run_id")

    if cfg.wandb.log:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            id=wandb_run_id,
            resume="allow",
        )

    ctx = setup_environment(cfg)
    train_loader, val_loader = prepare_dataloaders(cfg)

    model = create_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )

    iter_num = 0
    best_val_loss = 1e9
    if checkpoint:
        unwrapped_state_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in checkpoint["model"].items()
        }
        model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    if cfg.environment.compile:
        print("compiling the model..")
        model = torch.compile(model)

    print(
        f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M"
    )

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        ctx=ctx,  # type: ignore
        iter_num=iter_num,
        best_val_loss=best_val_loss,
    )

    if cfg.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
