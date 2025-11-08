"""
Transformer Implementation

A clean implementation of a transformer model with all components.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import einops
from dataclasses import dataclass
import math

from flow.utils import Config


def generic_layer_test_with_shapes(layer_class, config, input_shapes, input_types=None, expected_shape=None, expected_output=None, compare_func=None, tol=1e-5):
    """
    Tests a given layer with specified config and input SHAPES, checks output against expected_output.
    """
    try:
        # Make input tensors according to shape and type
        xs = []
        # Broadcast input_types if needed
        if input_types is None:
            input_types = ["randn"] * len(input_shapes)
        elif isinstance(input_types, str):
            input_types = [input_types] * len(input_shapes)
        elif len(input_types) == 1 and len(input_shapes) > 1:
            input_types = input_types * len(input_shapes)
        
        for shape, typ in zip(input_shapes, input_types):
            if typ == "randn":
                xs.append(torch.randn(*shape))
            elif typ == "rand":
                xs.append(torch.rand(*shape))
            elif typ == "int":
                xs.append(torch.randint(0, 10, shape))
            elif typ == "onehot":
                N = shape[-1]
                idxs = torch.randint(0, N, shape[:-1])
                oh = torch.nn.functional.one_hot(idxs, num_classes=N).float()
                xs.append(oh)
            else:
                raise ValueError(f"Unknown input_type: {typ}")

        # Instantiate the layer
        from dataclasses import fields, is_dataclass
        if is_dataclass(config):
            layer = layer_class(config)
            config_str = {f.name: getattr(config, f.name) for f in fields(config)}
        elif isinstance(config, dict):
            layer = layer_class(**config)
            config_str = config
        else:
            layer = layer_class(config)
            config_str = str(config)

        # Run the forward pass
        output = layer(*xs)
        print(f"[PASS] Layer {layer_class.__name__} ran successfully with config {config_str}. Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
    except Exception as e:
        print(f"[FAIL] Layer {layer_class.__name__} failed during forward with config {config}: {e}")
        return

    if expected_shape is not None:
        if output.shape != expected_shape:
            print(f"[FAIL] Output shape {output.shape} does not match expected shape {expected_shape}")
            return
        else:
            print(f"[PASS] Output shape {output.shape} matches expected shape {expected_shape}")

    if expected_output is not None:
        try:
            if compare_func is not None:
                comparison = compare_func(output, expected_output)
            else:
                comparison = torch.allclose(output, expected_output, atol=tol, rtol=tol)
            if comparison:
                print(f"[PASS] Output matches expected value.")
            else:
                print(f"[FAIL] Output does NOT match expected value.")
        except Exception as e:
            print(f"[FAIL] Error in output comparison: {e}")

class TimestepEmbedder(nn.Module): # taken from DiT Meta
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.frequency_embedding_dim, config.frequency_embedding_dim, bias=True),
            nn.GELU(),
            nn.Linear(config.frequency_embedding_dim, config.embed_dim, bias=True),
        )
        self.scale = config.timestep_scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: The max_period parameter controls the *minimum frequency* used in the embeddings. 
                           A larger max_period means the smallest frequency represented is lower. 
                           (Larger max_period increases the timescale range, so embeddings change more slowly 
                           for large timestep values. This is analogous to rotary embeddings' "base" or frequency granularity.)
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.scale, self.config.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb

class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.embed_dim))
        self.bias = nn.Parameter(torch.zeros(config.embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        # x : (batch, seq, embed_dim)
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)

class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.linear2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x : (batch, seq, embed_dim)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention. Note that we do not use causal masking here.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # multi-head attention
        self.Wq = nn.Parameter(torch.zeros(config.num_heads, config.embed_dim, config.head_dim))
        self.bq = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        self.Wk = nn.Parameter(torch.zeros(config.num_heads, config.embed_dim, config.head_dim))
        self.bk = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        self.Wv = nn.Parameter(torch.zeros(config.num_heads, config.embed_dim, config.head_dim))
        self.bv = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        nn.init.normal_(self.Wq, std=0.02)
        nn.init.normal_(self.Wk, std=0.02)
        nn.init.normal_(self.Wv, std=0.02)
        self.Wo = nn.Parameter(torch.zeros(config.num_heads, config.head_dim, config.embed_dim))
        self.bo = nn.Parameter(torch.zeros(config.embed_dim))
        nn.init.normal_(self.Wo, std=0.02)

    def forward(self, x):
        # x : (batch, seq, embed_dim)
        q = torch.einsum('bsd,ndh->bsnh', x, self.Wq) + self.bq  # (batch, seq, num_heads, head_dim)
        k = torch.einsum('bsd,ndh->bsnh', x, self.Wk) + self.bk  # (batch, seq, num_heads, head_dim)
        v = torch.einsum('bsd,ndh->bsnh', x, self.Wv) + self.bv  # (batch, seq, num_heads, head_dim)

        # compute attention weights
        attn_weights = torch.einsum('bsnh,btnh->bnst', q, k)  # (batch, num_heads, query_pos, key_pos)
        attn_weights /= self.config.head_dim ** 0.5
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # not causal
        after = torch.einsum('bnst,btnh->bsnh', attn_weights, v)  # (batch, query_pos, num_heads, head_dim)
        out = torch.einsum('bsnh,nhd->bsd', after, self.Wo) + self.bo
        return out


class TransformerLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layer_norm1 = LayerNorm(config)
        self.attention = Attention(config)
        self.layer_norm2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # x : (batch, seq, embed_dim)
        y = self.layer_norm1(x)
        y = self.attention(y)
        x = x + y   # residual connection
        y = self.layer_norm2(x)
        y = self.mlp(y)
        x = x + y   # residual connection
        return x


class Transformer(nn.Module):
    """Complete transformer model."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_input_tokens, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.context_len, config.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.unembedding = nn.Linear(config.embed_dim, config.output_channels * config.num_tokens)
        self.final_layer_norm = LayerNorm(config)

    def forward(self, x):
        # x : (batch, seq)
        emb = self.embedding(x) # (batch, seq, embed_dim)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1) # (batch, seq)
        pos_emb = self.positional_embedding(positions) # (batch, seq, embed_dim)
        x = pos_emb + emb # (batch, seq, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer_norm(x)
        logits = self.unembedding(x) # (batch, seq, num_tokens)
        if self.config.output_channels == 1:
            return logits
        else:
            return logits.reshape(x.shape[0], x.shape[1], self.config.output_channels, self.config.num_tokens)

class TimeAwareTransformer(nn.Module):
    """Transformer model that incorporates time information."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_input_tokens, config.embed_dim)
        self.timestep_embedder = TimestepEmbedder(config)
        self.positional_embedding = nn.Embedding(config.context_len, config.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.unembedding = nn.Linear(config.embed_dim, config.output_channels * config.num_tokens)
        self.final_layer_norm = LayerNorm(config)

    def forward(self, x, t):
        # x : (batch, seq)
        # t : (batch,)
        emb = self.embedding(x) # (batch, seq, embed_dim)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1) # (batch, seq)
        pos_emb = self.positional_embedding(positions) # (batch, seq, embed_dim)
        t_emb = self.timestep_embedder(t) # (batch, embed_dim)
        t_emb = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1) # (batch, seq, embed_dim)
        x = pos_emb + emb + t_emb # (batch, seq, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer_norm(x)
        logits = self.unembedding(x) # (batch, seq, output_channels * num_tokens)
        # add channels so it works with Gat flow matching
        if self.config.output_channels == 1:
            return logits
        else:
            return logits.reshape(x.shape[0], x.shape[1], self.config.output_channels, self.config.num_tokens)

class IgnorantTransformer(nn.Module):
    """Wrapper around a Transformer model that ignores time information."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)

    def forward(self, x, t=None):
        # x : (batch, seq)
        return self.transformer(x)

class SmallModel(nn.Module):
    """
    A small model for testing if things work
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.num_input_tokens, self.config.embed_dim)
        self.l1 = nn.Linear(self.config.embed_dim, self.config.embed_dim)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(self.config.embed_dim, self.config.num_tokens)
        self.time_embed = TimestepEmbedder(config)
        
    
    def forward(self, x, t):
        # x : (bs, s)
        # t : (bs)
        x = self.embedding(x) # (bs, s, d)
        t_embed = self.time_embed(t).unsqueeze(1).repeat(1, x.shape[1], 1) # (bs, s, d)
        x += t_embed
        x = self.l2(self.act(self.l1(x)))
        return x # (bs, s, d)

def test_all_components():
    """Test all transformer components."""
    config = Config()
    debug_config = Config(debug=True)
    
    print("Testing transformer components...")
    
    # Test individual components
    generic_layer_test_with_shapes(LayerNorm, config, [(2, 3, config.embed_dim)], "randn", expected_shape=(2, 3, config.embed_dim))
    generic_layer_test_with_shapes(MLP, config, [(2, 3, config.embed_dim)], "randn", expected_shape=(2, 3, config.embed_dim))
    generic_layer_test_with_shapes(Attention, config, [(2, 3, config.embed_dim)], "randn", expected_shape=(2, 3, config.embed_dim))
    generic_layer_test_with_shapes(TransformerLayer, config, [(2, 3, config.embed_dim)], "randn", expected_shape=(2, 3, config.embed_dim))
    generic_layer_test_with_shapes(Transformer, config, [(2, 3)], "int", expected_shape=(2, 3, config.num_tokens))
    
    print("\nTesting with debug config...")
    generic_layer_test_with_shapes(Transformer, debug_config, [(2, 3)], "int", expected_shape=(2, 3, config.num_tokens))


if __name__ == "__main__":
    test_all_components()
