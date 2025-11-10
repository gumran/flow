"""
Implementations of discrete flows like Campbell et al.

Using masked and uniform noises, and stochasticity defined in paper.
"""

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from dataclasses import dataclass

from flow.transformer import SmallModel
from flow.utils import Config, get_top_k_logits

# %%

class MaskedFMModel(nn.Module):
    def __init__(self, config: Config, model, mask_token_id = None):
        super().__init__()
        self.config = config
        self.model = model
        # num_input_tokens is the number of tokens in the vocabulary, including the mask token
        self.mask_token_id = mask_token_id or self.config.num_input_tokens - 1
        # the model predicts the logits for tokens

    def sample_t(self, x1, t):

        # x1 : (bs, c) or (...)
        # t : (bs,) or (1,) or (...)

        # make t have correct shape (bs,)
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x1.shape[0])
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.expand(x1.shape[0])
            else:
                assert t.shape[0] == x1.shape[0], f"t.shape: {t.shape}, x1.shape: {x1.shape}"
        else:
            raise ValueError(f"Invalid t shape: {t.shape}")

        t = t.to(x1.device)

        # the marginal distribution should be t * p_x1(x) + (1 - t) * p_mask(x)
        mask = (torch.rand_like(x1.float()) < (1 - t[:, None].expand_as(x1)))
        xt = torch.where(mask, self.mask_token_id, x1)
        return xt

    def get_train_loss(self, x):
        t = torch.rand(x.shape[0], device = x.device) # (bs,)
        xt = self.sample_t(x, t)
        logits = self.model(xt, t)
        target = x.clone()
        target[x == xt] = -1
        # transpose so cross entropy loss works
        loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-1, reduction='mean') # don't calculate loss on unmasked
        return loss

    def sample(self, bs, eta = None, dt = None, temperature = None, top_k = None, t = 0):
        if eta is None:
            eta = getattr(self.config, "eta", 0.0)
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        if temperature is None:
            temperature = getattr(self.config, "temperature", 1.0)
        if top_k is None:
            top_k = getattr(self.config, "top_k", None)
        # eta is stochasticity
        # the rate matrix is 1/(1 - t) from mask to unmasked
        # stochasticity eta is the rate matrix of given element back to mask
        # Handle both batch size (int) and initial tensor
        if isinstance(bs, torch.Tensor):
            x = bs.clone()
            bs = x.shape[0]
        else:
            c = self.config.context_len
            x = torch.full((bs, c), self.mask_token_id, device = self.config.device) # initial input
        while t < 1:
            logits = self.model(x, torch.full((bs,), t, device=self.config.device)) # (bs, c, s)
            if top_k is not None:
                logits = get_top_k_logits(logits, top_k)
            unmasked = Categorical(logits=logits / temperature).sample()
            # probability of unmasked going back to mask is eta * dt
            # probability of mask going to unmasked is 1/(1 - t) * dt + eta t/(1 - t) * dt
            to_unmask = (torch.rand_like(x.float()) < (1 + eta * t) * dt / (1 - t)) \
                    & (x == self.mask_token_id)
            to_mask = (torch.rand_like(x.float()) < eta * dt) & (x != self.mask_token_id)
            x = torch.where(to_unmask, unmasked, x)
            t += dt
            if t < 1:
                x = torch.where(to_mask, self.mask_token_id, x)
            else:
                # unmask all
                to_unmask = (x == self.mask_token_id)
                x = torch.where(to_unmask, unmasked, x)
        return x

# %%

def test_masked_model():
    config = Config(num_tokens=20, num_input_tokens=21)
    model = SmallModel(config).to(config.device)
    masked_model = MaskedFMModel(config, model)
    x1 = torch.randint(0, config.num_tokens, (10, 10))
    # t = torch.rand((10,))
    t = torch.tensor(0.5)
    xt = masked_model.sample_t(x1, t)
    print(xt)

    out = masked_model.sample(15)
    print(out)
    print(out.shape)

# %%

if __name__ == "__main__":
    test_masked_model()

# %%

class UniformFMModel(nn.Module):
    def __init__(self, config: Config, model):
        super().__init__()
        self.config = config
        self.model = model

    def sample_t(self, x1, t):

        # x1 : (bs, c) or (...)
        # t : (bs,) or (1,) or (...)

        # make t have correct shape (bs,)
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x1.shape[0])
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.expand(x1.shape[0])
            else:
                assert t.shape[0] == x1.shape[0], f"t.shape: {t.shape}, x1.shape: {x1.shape}"
        else:
            raise ValueError(f"Invalid t shape: {t.shape}")

        t = t.to(x1.device)

        x0 = torch.randint_like(x1, self.config.num_tokens)
        # the marginal distribution should be t * p_1(x) + (1 - t) * p_0(x)
        mask = (torch.rand_like(x1.float()) < (1 - t[:, None].expand_as(x1)))
        xt = torch.where(mask, x0, x1)
        return xt

    def get_train_loss(self, x):
        t = torch.rand(x.shape[0], device = x.device) # (bs,)
        xt = self.sample_t(x, t)
        logits = self.model(xt, t)
        # transpose so cross entropy loss works
        loss = F.cross_entropy(logits.transpose(1, 2), x, reduction='mean')
        return loss

    def sample(self, bs, eta = None, dt = None, temperature = None, top_k = None):
        if eta is None:
            eta = getattr(self.config, "eta", 0.0)
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        if temperature is None:
            temperature = getattr(self.config, "temperature", 1.0)
        if top_k is None:
            top_k = getattr(self.config, "top_k", None)
        # eta is stochasticity
        # the rate matrix is 1/(1 - t) from mask to unmasked
        # stochasticity eta is the rate matrix of given element back to mask
        # Handle both batch size (int) and initial tensor
        if isinstance(bs, torch.Tensor):
            x = bs.clone()
            bs = x.shape[0]
        else:
            c = self.config.context_len
            x = torch.randint(0, self.config.num_tokens, (bs, c), device=self.config.device) # initial input
        s = self.config.num_tokens
        t = 0
        while t < 1:
            logits = self.model(x, torch.full((bs,), t, device=self.config.device)) # (bs, c, s)
            if top_k is not None:
                logits = get_top_k_logits(logits, top_k)
            pred_tokens = Categorical(logits=logits / temperature).sample()
            rand_tokens = torch.randint_like(pred_tokens, self.config.num_tokens)
            # if we are at pred token, the prob of going to random token is eta * dt
            # flow between pred and a random other token is then eta * (t + (1-t)/S)
            # probability of going to pred token is 1/(1 - t) * dt + eta * (St/(1 - t) + 1) * dt if we are not at pred_token
            to_unmask = (torch.rand_like(x.float()) < (1 + eta * s * t) * dt / (1 - t) + eta * dt) \
                    & (x != pred_tokens)
            # in to_mask, we multiply by s, since then for each potential token j,
            # the probability of masking to j is
            # p(mask happens) = eta * s * dt
            # multiplied by p(rand_token = j) = 1/s
            # so overall p(masked to j) = eta * dt
            to_mask = (torch.rand_like(x.float()) < eta * s * dt) & (x == pred_tokens)
            x = torch.where(to_unmask, pred_tokens, x)
            t += dt
            if t < 1:
                x = torch.where(to_mask, rand_tokens, x)
            else:
                # unmask all
                to_unmask = (x != pred_tokens)
                x = torch.where(to_unmask, pred_tokens, x)
        return x

# %%

def test_uniform_model():
    config = Config(num_tokens=20)
    model = SmallModel(config).to(config.device)
    masked_model = UniformFMModel(config, model)
    x1 = torch.randint(0, config.num_tokens, (10, 10))
    # t = torch.rand((10,))
    t = torch.tensor(0.5)
    xt = masked_model.sample_t(x1, t)
    print(xt)

    out = masked_model.sample(15)
    print(out)
    print(out.shape)

# %%

if __name__ == "__main__":
    test_uniform_model()

# %%
