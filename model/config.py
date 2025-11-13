from dataclasses import dataclass, field
from typing import List, Literal, Optional


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    norm_qk_type: Literal["default", "olmo2"] = "default"
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: Optional[int] = None
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    attn_bias: bool = False
    attention_scores_scalar: Optional[int] = None
    sliding_window_size: Optional[int] = None
    sliding_window_indices: Optional[List] = None
    # if `attention_logit_softcapping` is used, cannot use optimized
    # `torch.nn.functional.scaled_dot_product_attention` (which implements
    # Flash attention), may result in higher memory and runtime footprint.
    attention_logit_softcapping: Optional[float] = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: Optional[dict] = None
    # Transformer block (MLP)
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    bias: bool = True
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = (
        "GptNeoxMLP"
    )
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_expert_per_token: int = 0
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: Optional[float] = None
    norm_1: bool = True
    norm_2: bool = True
    latent_attention: Optional[dict] = None
    # The base period of the RoPE embeddings for local attention.
    # If not provided, rope_theta will be used for both local and global attention.
    rope_local_base_freq: Optional[float] = None
    rope_indices: Optional[List] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(
                    f"The config {self.name!r}, needs to set the `intermediate_size`"
                )
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None and self.sliding_window_indices is None:
            self.sliding_window_indices = [1] * self.n_layer

        if self.rope_local_base_freq is not None and self.rope_indices is None:
            self.rope_indices = [1] * self.n_layer

        if self.latent_attention is not None:
            self.q_lora_rank = self.latent_attention.get("q_lora_rank")
            self.kv_lora_rank = self.latent_attention.get("kv_lora_rank")
            self.qk_rope_head_dim = self.latent_attention.get("qk_rope_head_dim")
            self.qk_nope_head_dim = self.latent_attention.get("qk_nope_head_dim")
            self.v_head_dim = self.latent_attention.get("v_head_dim")
            assert (
                self.q_lora_rank
                and self.kv_lora_rank
                and self.qk_rope_head_dim
                and self.qk_nope_head_dim
                and self.v_head_dim
            ) is not None
            assert (
                self.n_head == self.n_query_groups
            ), "Latent attention does not support MQA/GQA"
            self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
            self.rope_n_elem = self.qk_rope_head_dim


@dataclass
class TRLMConfig(Config):
    H_cycles: int = 4
    L_cycles: int = 3
    halt_max_steps: int = 16
    no_ACT_continue: bool = False

    name: str = ""
    hf_config: dict = field(default_factory=dict)
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    norm_qk_type: Literal["default", "olmo2"] = "default"
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: Optional[int] = None
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    attn_bias: bool = False
    attention_scores_scalar: Optional[int] = None
    sliding_window_size: Optional[int] = None
    sliding_window_indices: Optional[List] = None
    # if `attention_logit_softcapping` is used, cannot use optimized
    # `torch.nn.functional.scaled_dot_product_attention` (which implements
    # Flash attention), may result in higher memory and runtime footprint.
    attention_logit_softcapping: Optional[float] = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: Optional[dict] = None
    # Transformer block (MLP)
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    bias: bool = True
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = (
        "GptNeoxMLP"
    )
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_expert_per_token: int = 0
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: Optional[float] = None
    norm_1: bool = True
    norm_2: bool = True
    latent_attention: Optional[dict] = None
    # The base period of the RoPE embeddings for local attention.
    # If not provided, rope_theta will be used for both local and global attention.
    rope_local_base_freq: Optional[float] = None
    rope_indices: Optional[List] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(
                    f"The config {self.name!r}, needs to set the `intermediate_size`"
                )
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None and self.sliding_window_indices is None:
            self.sliding_window_indices = [1] * self.n_layer

        if self.rope_local_base_freq is not None and self.rope_indices is None:
            self.rope_indices = [1] * self.n_layer

        if self.latent_attention is not None:
            self.q_lora_rank = self.latent_attention.get("q_lora_rank")
            self.kv_lora_rank = self.latent_attention.get("kv_lora_rank")
            self.qk_rope_head_dim = self.latent_attention.get("qk_rope_head_dim")
            self.qk_nope_head_dim = self.latent_attention.get("qk_nope_head_dim")
            self.v_head_dim = self.latent_attention.get("v_head_dim")
            assert (
                self.q_lora_rank
                and self.kv_lora_rank
                and self.qk_rope_head_dim
                and self.qk_nope_head_dim
                and self.v_head_dim
            ) is not None
            assert (
                self.n_head == self.n_query_groups
            ), "Latent attention does not support MQA/GQA"
            self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
            self.rope_n_elem = self.qk_rope_head_dim
