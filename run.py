import torch
from transformers import AutoTokenizer

from typing import List
from model.trlm import TRLM, TRLMCarry
from model.config import TRLMConfig

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--checkpoint-path", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--l-cycles", type=int, default=6)
parser.add_argument("--h-cycles", type=int, default=1)
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
prompt = args.prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

model_args = dict(
    n_layer=args.n_layer,
    n_head=6,
    n_embd=756,
    block_size=1024,
    bias=False,
    vocab_size=50304,
    H_cycles=args.h_cycles,
    L_cycles=args.l_cycles,
    halt_max_steps=1,
    no_ACT_continue=False,
)
gptconf = TRLMConfig(**model_args)

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def load_checkpoint(path: str):
    trlm: TRLM = TRLM(gptconf)
    trlm.load_state_dict(torch.load(path)["model"])
    trlm.to(device)
    trlm.eval()
    return trlm


def run(model: TRLM, prompts: List[str]):
    input_ids = tokenizer(prompts, add_special_tokens=False).to(device)
    batch = {"input_ids": input_ids}
    with torch.no_grad():
        carry = model.initial_carry(batch, device=device)
        carry, outputs = model(carry, batch)
        logits = outputs["logits"]
