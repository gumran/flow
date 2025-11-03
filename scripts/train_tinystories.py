"""
A small flow model for the TinyStories dataset.
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from flow.transformer import TimeAwareTransformer, Config
from flow.campbell_flow import MaskedFMModel, UniformFMModel
from flow.general_flow import GeneralFlow, WeightScheduler, DataSampler, NoiseSampler, LinearKappa

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

small_config = Config(
    num_tokens=len(tokenizer),
    embed_dim=128,
    mlp_dim=256,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=16,
    context_len=384,
    num_layers=8,
    timestep_scale=1000.0,
    output_channels=2,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
config = small_config
large_config = Config(
    num_tokens=len(tokenizer),
    embed_dim=512,
    mlp_dim=2048,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=64,
    context_len=384,
    num_layers=17,
    timestep_scale=1000.0,
    output_channels=2,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
# %%

print(tokenizer.encode("Hello, world!"))
print(tokenizer.decode(tokenizer.encode("Hello, world!")))
# %%

ds = load_dataset("roneneldan/TinyStories")
print(ds)
print(ds['train'][0])
print(tokenizer.encode(ds['train'][0]['text']))
print(tokenizer.decode(tokenizer.encode(ds['train'][0]['text'])))
# %%

print(ds['train'][1])
print(tokenizer.encode(ds['train'][1]['text']))
print(tokenizer.decode(tokenizer.encode(ds['train'][1]['text'])))
# %%

print(ds['train'][2])
print(tokenizer.encode(ds['train'][2]['text']))
print(tokenizer.decode(tokenizer.encode(ds['train'][2]['text'])))

# %%

# check number of tokens
for i in range(100):
    print(len(tokenizer.encode(ds['train'][i]['text'])))
# %%

def make_tinystories_dataloader(batch_size=8, num_proc=4, subset_size=None, tokenizer_name="roberta-base", block_size=384):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        # safe fallback for models like GPT-2
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size)

    ds = ds.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=["text"])
    if subset_size:
        ds = ds.select(range(subset_size))

    def collate_fn(batch):
        # convert list of dicts → dict of stacked tensors
        return {k: torch.stack([torch.tensor(x[k]) for x in batch]) for k in batch[0]}

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# %%

# test the dataloader
dataloader = make_tinystories_dataloader(batch_size=8, num_proc=4)
for batch in dataloader:
    print(batch)
    break

# %%

# ok now time for a small train (to see overfitting)
config = large_config
batch_size = 16
dataloader = make_tinystories_dataloader(batch_size=batch_size, num_proc=4, subset_size=100)
mask_token_id = tokenizer.mask_token_id
model = TimeAwareTransformer(config)
model.to(config.device)
kappa1 = LinearKappa(config)
weight_scheduler = WeightScheduler(config, kappa1) # 2 kappas
samplers = [
    DataSampler(config),
    NoiseSampler(config)
]
gf = GeneralFlow(config, model, samplers, weight_scheduler)
print("num params:", sum(p.numel() for p in model.parameters()))
# %%
# train the model

# early stopping
# best_loss = float('inf')
patience = 5_000
# epochs_no_improve = 0
min_delta = 1e-4  # how small an improvement must be to count

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# num_epochs = 250
total_steps = 150_000
# num_training_steps = num_epochs * len(dataloader)
num_warmup_steps = int(0.05 * total_steps)  # 5% warmup
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, smoothing=0.1)

step = 0
while step < total_steps:
    for batch in dataloader:
        if step >= total_steps:
            break

        step += 1
        optimizer.zero_grad(set_to_none=True)

        # Forward / backward
        x1 = batch["input_ids"].to(config.device)
        x0 = torch.full_like(x1, mask_token_id)
        t  = torch.rand(x1.shape[0], device=config.device)

        loss = gf.get_train_loss(x0, x1, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Tracking
        running_loss += loss.item()
        avg_loss = running_loss / step
        lr = scheduler.get_last_lr()[0]
        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_resv = torch.cuda.memory_reserved() / 1024**2

        # tqdm update
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "lr": f"{lr:.2e}",
            "mem": f"{mem_alloc:.0f}/{mem_resv:.0f} MB"
        })

        # Early stopping
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        if steps_no_improve >= patience:
            pbar.write(f"\nEarly stopping at step {step} | best_loss={best_loss:.4f}")
            step = total_steps  # force exit
            break
pbar.close()
print(f"Training complete at step {step:,} | best_loss={best_loss:.4f}")
    # %%

# create some examples

x0 = torch.full((1, 384), mask_token_id, device=config.device)
result = gf.forward_sample(x0, dt = 1e-4)
print(tokenizer.decode(result[0].cpu().numpy()))
# %%
x0 = torch.full((1, 384), mask_token_id, device=config.device)
result = gf.corrector_sample(x0, dt = 1e-4)
print(tokenizer.decode(result[0].cpu().numpy()))

# %%
