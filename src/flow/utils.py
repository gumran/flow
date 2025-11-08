# %%

import torch
from dataclasses import dataclass

@dataclass
class Config:
    num_tokens: int = 20                        # number of tokens in the vocabulary
    embed_dim: int = 16                         # dimension of the embedding
    mlp_dim: int = 32                           # dimension of the MLP
    frequency_embedding_dim: int = 32           # dimension of the frequency embedding
    num_heads: int = 4                          # number of heads in the attention
    head_dim: int = 4                           # dimension of each head
    context_len: int = 1024                     # maximum length of the context
    num_layers: int = 6                         # number of layers in the transformer
    timestep_scale: float = 1000.0              # how much to scale the timestep embedding
    debug: bool = False                         # whether to print debug information
    device: str = "cuda" \
        if torch.cuda.is_available() \
        else "cpu"                              # device to use
    num_input_tokens: int = None                # number of input tokens, can be different from num_tokens (useful for masking, etc.)
    output_channels: int = 1                    # number of channels in the output
    add_residual: bool = False                  # add a residual kappa
    debug: bool = False                         # debug readouts
    seed: int = 42                             # seed for reproducibility

    def __post_init__(self):
        if self.num_input_tokens is None:
            self.num_input_tokens = self.num_tokens

@dataclass
class SmallConfig(Config):
    num_tokens: int = 128
    embed_dim: int = 256
    mlp_dim: int = 128
    frequency_embedding_dim: int = 128
    num_heads: int = 8
    head_dim: int = 16
    context_len: int = 384
    num_layers: int = 8
    timestep_scale: float = 1000.0
    output_channels: int = 1
    add_residual: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

@dataclass
class LargeConfig(Config):
    embed_dim: int = 512
    mlp_dim: int = 2048
    frequency_embedding_dim: int = 128
    num_heads: int = 8
    head_dim: int = 64
    context_len: int = 384
    num_layers: int = 16
    timestep_scale: float = 1000.0
    output_channels: int = 1
    add_residual: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def get_top_k_logits(logits, top_k):
    values, indices = torch.topk(logits, top_k, dim=-1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    masked_logits = torch.where(mask, logits, -torch.inf)
    return masked_logits

# %%

if __name__ == "__main__":

    logits = torch.randn(5, 10, 10)
    print(logits[0, 0, :])
    top_k_logits = get_top_k_logits(logits, 3)
    print(top_k_logits[0, 0, :])

# %%
