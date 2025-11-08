"""
Implement empirical DFM flows for testing and inference.
"""

# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import einops
from torch.distributions import Categorical
from flow.utils import Config

def display_decoded_tokens(tensor, tokenizer, title=None):
    """
    Nicely display a tensor of tokens, decode with the given tokenizer, removing <s> and <pad>,
    and replacing <mask> tokens with the unicode replacement character (boxed question mark, U+FFFD).
    Args:
        tensor (torch.Tensor): Tensor of token ids. Shape: (N, sequence_length) or (sequence_length,)
        tokenizer: The tokenizer to decode with.
        title (str, optional): Optional title to display above results.
    """
    def postprocess(text):
        # Remove <s>, <pad>, and </s>
        for special in ["<s>", "<pad>", "</s>"]:
            text = text.replace(special, "")
        # Replace <mask> with unicode replacement character (�, U+FFFD)
        mask_str = None
        if hasattr(tokenizer, "mask_token") and tokenizer.mask_token is not None:
            mask_str = tokenizer.mask_token
        else:
            mask_str = "<mask>"
        text = text.replace(mask_str, "\uFFFD")
        # Remove leading/trailing whitespace and collapse multiple spaces
        return " ".join(text.strip().split())

    if title:
        print(f"== {title} ==")
    if tensor.ndim == 1:
        print(postprocess(tokenizer.decode(tensor)))
    else:
        for i, row in enumerate(tensor):
            print(f"[{i:2d}]: {postprocess(tokenizer.decode(row))}")

# %%

class EmpiricalDFM(nn.Module):
    def __init__(self, config: Config, dataset : DataLoader, mask_token_id = None, initial_type = "mask", tokenizer = None):
        super().__init__()
        self.config = config
        self.dataset = dataset # tokens, (bs, c)
        self.mask_token_id = mask_token_id
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.initial_type = initial_type

    def calculate_masked_probabilities(self, input_tokens: torch.Tensor, t: torch.Tensor = None):
        # input_tokens : (bs, c) or (...)
        # t : (bs,) or (1,) or (...)
        assert self.mask_token_id is not None, "mask_token_id must be set"

        # for each batch in input_tokens,
        # we go through each each dataset example
        # and see if it fits the mask
        # if it does, we add its tokens to its predicted flow
        # and increment its number of samples by 1
        # at the end we divide the predicted flow by the number of samples
        # to get the average token
        bs, c = input_tokens.shape
        device = input_tokens.device

        is_unmasked = (input_tokens != self.mask_token_id)
        is_masked = ~is_unmasked

        num_samples = torch.zeros(bs, dtype=torch.float32, device=device)
        predicted_flow = torch.zeros((bs, c, self.config.num_tokens), dtype=torch.float32, device=device)
        
        for dataset_batch in self.dataset:
            if isinstance(dataset_batch, dict):
                dataset_tokens = dataset_batch['input_ids']
            else:
                dataset_tokens = dataset_batch
            assert type(dataset_tokens) == torch.Tensor, "dataset_tokens must be a tensor"
            dataset_tokens = dataset_tokens.to(device)
            N = dataset_tokens.shape[0]

            # (N, bs, c)
            matches = (dataset_tokens.unsqueeze(1) == input_tokens.unsqueeze(0)) | (~is_unmasked.unsqueeze(0))
            match_mask = matches.all(dim=2)  # (N, bs)


            dataset_indices, batch_indices = match_mask.nonzero(as_tuple=True)
            if dataset_indices.numel() == 0:
                continue

            dataset_onehot = F.one_hot(dataset_tokens, num_classes=self.config.num_tokens) # (N, c, s)
            additional_flow = einops.einsum(match_mask.float(), dataset_onehot.float(), "n bs , n c s -> bs c s")
            predicted_flow += additional_flow

            num_samples += match_mask.sum(dim = 0)

            # matching_examples = dataset_tokens[dataset_indices]  # (M, c)
            # M = matching_examples.size(0)

            # # count matches per batch
            # num_samples.scatter_add_(0, batch_indices, torch.ones(M, device=device))

            # masked_mask = is_masked[batch_indices]  # (M, c)
            # batch_idx_expanded = batch_indices.unsqueeze(1).expand(-1, c) # (M, c)
            # pos_idx = torch.arange(c, device=device).unsqueeze(0).expand(M, -1) # (M, c)

            # flat_mask = masked_mask.flatten() # (M*c,)
            # flat_batch = batch_idx_expanded.flatten()[flat_mask] # (which batch to target)
            # flat_pos = pos_idx.flatten()[flat_mask] # (which position to target)
            # flat_tokens = matching_examples.flatten()[flat_mask].long() # (which token, from the corresponding matching token)

            # # Use scatter_add_ to properly accumulate when there are duplicate indices
            # # Flatten indices: (batch, pos, token) -> linear index
            # linear_indices = (flat_batch * c * self.config.num_tokens + 
            #                 flat_pos * self.config.num_tokens + 
            #                 flat_tokens)
            # predicted_flow.view(-1).scatter_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.float32))

        valid = num_samples > 0
        predicted_flow[valid] /= num_samples[valid].view(-1, 1, 1)
        predicted_flow[~valid] = 0
        return predicted_flow

    def calculate_uniform_probabilities(self, input_tokens: torch.Tensor, t: torch.Tensor):
        """
        Calculate the uniform probabilities of the input tokens.
        Handles small probabilities by using a log-space accumulator.
        """
        # input_tokens : (bs, c) or (...)
        # t : (bs,) or (1,) or (...)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != input_tokens.shape[0]:
            t = t.expand(input_tokens.shape[0])

        bs, c = input_tokens.shape
        device = input_tokens.device
        T = self.config.num_tokens

        t = t.to(device)

        # need to use float64 as we have small probabilities
        # num_samples = torch.zeros(bs, dtype=torch.float64, device=device)
        # predicted_flow = torch.zeros((bs, c, self.config.num_tokens), dtype=torch.float64, device=device)

        # precompute logs (per bs)
        log_a = torch.log(t + (1 - t) / T)          # shape (bs,)
        log_b = torch.log((1 - t) / T)              # shape (bs,)

        # running accumulators (global scale per bs)
        m = torch.full((bs,), float('-inf'), dtype=torch.float64, device=device)
        num_scaled = torch.zeros((bs, c, T), dtype=torch.float64, device=device)
        den_scaled = torch.zeros((bs,), dtype=torch.float64, device=device)

        for dataset_batch in self.dataset:
            if isinstance(dataset_batch, dict):
                dataset_tokens = dataset_batch['input_ids']
            else:
                dataset_tokens = dataset_batch
            assert type(dataset_tokens) == torch.Tensor, "dataset_tokens must be a tensor"
            dataset_tokens = dataset_tokens.to(device)
            N = dataset_tokens.shape[0]

            # (N, bs, c)
            matches = (dataset_tokens.unsqueeze(1) == input_tokens.unsqueeze(0))
            match_mask = matches.sum(dim=2)  # (N, bs), number of matches

            # at time t, if a given example matches in K places
            # out of C, and there are T tokens in total, then
            # the probability of this happening is
            # (t + (1-t)/T)**(K) * ((1-t)/T) ** (C - K)
            logp = match_mask * log_a.view(1, bs) + (c - match_mask) * log_b.view(1, bs)  # (N, bs)
            # probability = (t + (1-t)/T)**(match_mask.to(torch.float64)) \
            #    * ((1-t)/T)**(c - match_mask.to(torch.float64))

            # per-bs max for this block
            m_batch = logp.max(dim=0).values  # (bs,)
            # combine scales
            m_new = torch.maximum(m, m_batch)  # (bs,)

            # rescale old accumulators to new scale
            alpha = torch.exp(m - m_new)                  # (bs,), less than 1
            num_scaled *= alpha.view(bs, 1, 1)             # (bs, c, T)
            den_scaled *= alpha

            # add this block's contributions at the new scale
            # beta[n,b] = exp(logp[n,b] - m_new[b])
            beta = torch.exp(logp - m_new.view(1, bs))    # (N, bs)

            dataset_onehot = F.one_hot(dataset_tokens, num_classes=self.config.num_tokens) # (N, c, s)
            num_scaled += einops.einsum(beta, dataset_onehot.to(torch.float64), "n bs , n c s -> bs c s")

            den_scaled += beta.sum(dim = 0)

            m = m_new

        flow = torch.zeros_like(num_scaled, dtype=torch.float64)
        valid = den_scaled > 0
        flow[valid] = num_scaled[valid] / den_scaled[valid].view(-1, 1, 1)
        # if desired, zero-out invalid
        flow[~valid] = 0

        return flow.to(torch.float32)

        # valid = num_samples > 0
        # predicted_flow[valid] /= num_samples[valid].view(-1, 1, 1)
        # predicted_flow[~valid] = 0
        # return predicted_flow.to(torch.float32)

    def calculate_probabilities(self, input_tokens: torch.Tensor, t: torch.Tensor):

        if self.initial_type == "mask":
            probabilities = self.calculate_masked_probabilities(input_tokens, t)
        elif self.initial_type == "uniform":
            probabilities = self.calculate_uniform_probabilities(input_tokens, t)
        else:
            raise ValueError(f"Invalid initial type: {self.initial_type}")

        # sometimes probabilities are all 0. In this case,
        # just keep the token as it is

        one_hot = F.one_hot(input_tokens, num_classes=self.config.num_tokens) # (bs, c, s)
        zero_probabilities = (probabilities.sum(dim = 2) == 0) # (bs, c)
        zero_idxs = zero_probabilities.nonzero(as_tuple=True)
        # print(zero_idxs)
        probabilities[zero_idxs] = one_hot[zero_idxs].float()
        return probabilities

    def sample(self, bs, eta = None, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        if eta is None:
            eta = getattr(self.config, "eta", 0.0)
        c = self.config.context_len
        s = self.config.num_tokens
        if self.initial_type == "mask":
            x = torch.full((bs, c), self.mask_token_id, device = self.config.device) # initial input
        elif self.initial_type == "uniform":
            x = torch.randint(0, self.config.num_tokens, (bs, c), device = self.config.device)
        else:
            raise ValueError(f"Invalid initial type: {self.initial_type}")
        t = 0
        i = 0
        while t < 1:
            probabilities = self.calculate_probabilities(x, torch.full((bs,), t, device=self.config.device))
            pred_tokens = Categorical(probs=probabilities).sample()
            if self.initial_type == "mask":
                other_tokens = torch.full((bs, c), self.mask_token_id, device = self.config.device)
                to_unmask = (torch.rand_like(x.float()) < (1 + eta * t) * dt / (1 - t)) \
                        & (x == self.mask_token_id)
                to_mask = (torch.rand_like(x.float()) < eta * dt) & (x != self.mask_token_id)
                x = torch.where(to_unmask, pred_tokens, x)
            elif self.initial_type == "uniform":
                other_tokens = torch.randint_like(x, self.config.num_tokens)
                to_unmask = (torch.rand_like(x.float()) < (1 + eta * s * t) * dt / (1 - t) + eta * dt) \
                        & (x != pred_tokens)
                to_mask = (torch.rand_like(x.float()) < eta * s * dt) & (x == pred_tokens)
                x = torch.where(to_unmask, pred_tokens, x)
            else:
                raise ValueError(f"Invalid initial type: {self.initial_type}")
            t += dt
            if t < 1:
                x = torch.where(to_mask, other_tokens, x)
            else:
                # unmask all
                if self.initial_type == "mask":
                    to_unmask = (x == self.mask_token_id)
                elif self.initial_type == "uniform":
                    to_unmask = (x != pred_tokens)
                x = torch.where(to_unmask, pred_tokens, x)
            if i % 10 == 0:
                display_decoded_tokens(x, self.tokenizer, "Sampled tokens")
            i += 1
        return x
    
    def one_at_a_time_sample(self, bs, eta = None):
        if eta is None:
            eta = getattr(self.config, "eta", 0.0)
        c = self.config.context_len
        s = self.config.num_tokens
        x = torch.full((bs, c), self.mask_token_id, device = self.config.device) # initial input
        masked_tokens = c
        while masked_tokens > 0:
            probabilities = self.calculate_masked_probabilities(x)
            pred_tokens = Categorical(probs=probabilities).sample()
            # All batches have the same number (c) of masked tokens, simply pick a single random position to unmask
            masked_positions = (x == self.mask_token_id)  # (bs, c)
            # find all masked indices, for all batches these are the same
            masked_idxs = masked_positions.nonzero(as_tuple=True)[1].view(x.shape[0], -1)
            # pick a single random position to unmask (same across all batches)
            rand_pos = torch.randint(0, masked_idxs.shape[1], (1,))
            to_unmask = torch.zeros_like(x, dtype=torch.bool)
            to_unmask[torch.arange(x.shape[0]), masked_idxs[:, rand_pos].squeeze(1)] = True
            x = torch.where(to_unmask, pred_tokens, x)
            masked_tokens -= 1
            print(self.tokenizer.decode(x[0]))
        return x

# %%

dataset_path = "/scratch/inath/datasets/tinystories_1000_dataset"
dataset = load_from_disk(dataset_path).with_format("torch")
# %%

dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
# %%
for batch in dataloader:
    print(batch)
    break
# %%

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
mask_token_id = tokenizer.mask_token_id
example_tokens = dataset[[420, 69, 67]]['input_ids']
example_masked = example_tokens.clone()
example_masked = torch.where(
    torch.rand(example_masked.shape) < 0.993,
    mask_token_id,
    example_masked).to('cuda')


# %%

config = Config(
    num_tokens=tokenizer.vocab_size,
    context_len=384,
)



# %%


empirical_dfm = EmpiricalDFM(config, dataloader, mask_token_id)
masked_probs = empirical_dfm.calculate_masked_probabilities(example_masked)
print(masked_probs[0][1])
nonzero_indices = torch.nonzero(masked_probs[0][1], as_tuple=True)
print(nonzero_indices)
print(masked_probs[0][1][nonzero_indices])
max_index = masked_probs[0][1].argmax()
print(max_index)
print(tokenizer.decode(max_index)) # "Once"

# %%

example_uniformized = example_tokens.clone()
random_tokens = torch.randint(0, tokenizer.vocab_size, example_tokens.shape)
example_uniformized = torch.where(
    torch.rand(example_uniformized.shape) < 0.995,
    random_tokens,
    example_uniformized).to('cuda')


uniform_probs = empirical_dfm.calculate_uniform_probabilities(example_uniformized, torch.tensor([0.5]))
token_number = 3
print(uniform_probs[0][token_number])
nonzero_indices = torch.nonzero(uniform_probs[0][token_number], as_tuple=True)
print(nonzero_indices)
print(uniform_probs[0][token_number][nonzero_indices])
max_index = uniform_probs[0][token_number].argmax()
print(max_index)
print(tokenizer.decode(max_index)) # "Once"

# %%

empirical_dfm = EmpiricalDFM(config, dataloader, mask_token_id, initial_type="mask")
example_sampled = empirical_dfm.sample(1, dt=0.001)
display_decoded_tokens(example_sampled, tokenizer, "Sampled tokens")
# %%

empirical_dfm = EmpiricalDFM(config, dataloader, mask_token_id, initial_type="uniform")
example_sampled = empirical_dfm.sample(1, dt=0.001)
display_decoded_tokens(example_sampled, tokenizer, "Sampled tokens")
# %%

empirical_dfm = EmpiricalDFM(config, dataloader, mask_token_id, initial_type="mask")
example_sampled = empirical_dfm.one_at_a_time_sample(1)
print(tokenizer.decode(example_sampled[0]))
# %%
