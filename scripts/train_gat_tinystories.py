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
import wandb

from flow.transformer import IgnorantTransformer, Config
from flow.campbell_flow import MaskedFMModel, UniformFMModel
from flow.general_flow import GeneralFlow, WeightScheduler, DataSampler, NoiseSampler, LinearKappa, UsualFlow

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
    seed=42,
)
large_config = Config(
    num_tokens=len(tokenizer),
    embed_dim=512,
    mlp_dim=2048,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=64,
    context_len=384,
    num_layers=16,
    timestep_scale=1000.0,
    output_channels=2,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

config = large_config

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
# %%
def test_tokenizer():
    ds = load_dataset("roneneldan/TinyStories")
    print(tokenizer.encode("Hello, world!"))
    print(tokenizer.decode(tokenizer.encode("Hello, world!")))
    print(ds['train'][0])
    print(tokenizer.encode(ds['train'][0]['text']))
    print(tokenizer.decode(tokenizer.encode(ds['train'][0]['text'])))
    print(ds['train'][1])
    print(tokenizer.encode(ds['train'][1]['text']))
    print(tokenizer.decode(tokenizer.encode(ds['train'][1]['text'])))
    print(ds['train'][2])
    print(tokenizer.encode(ds['train'][2]['text']))
    print(tokenizer.decode(tokenizer.encode(ds['train'][2]['text'])))
# %%

def make_tinystories_dataloader(batch_size=8, num_proc=4, subset_size=None, tokenizer_name="roberta-base", block_size=384, save_path=None):
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
    if save_path is not None:
        ds.save_to_disk(save_path)
    def collate_fn(batch):
        # convert list of dicts → dict of stacked tensors
        return {k: torch.stack([torch.tensor(x[k]) for x in batch]) for k in batch[0]}

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# %%

# test the dataloader
def test_dataloader():
    dataloader = make_tinystories_dataloader(batch_size=8, num_proc=4)
    for batch in dataloader:
        print(batch)
        break

# %%

# ok now time for a small train (to see overfitting)
subset_size = None
# subset_size = 10
batch_size = 16
save_path = f"/scratch/inath/datasets/tinystories_{subset_size or 'full'}_dataset"
dataloader = make_tinystories_dataloader(
    batch_size=min(batch_size, subset_size or float('inf')),
    num_proc=4,
    subset_size=subset_size,
    save_path=save_path if subset_size is not None else None
    )
mask_token_id = tokenizer.mask_token_id
model = IgnorantTransformer(config)
model.to(config.device)
gf = UsualFlow(config, model)
print("num params:", sum(p.numel() for p in model.parameters()))
# %%
# train the model

# Initialize wandb
wandb.init(
    project="flow-tinystories",
    name=f"general_flow_{subset_size or 'full'}",
    config={
        "model": "UsualFlow",
        "subset_size": subset_size,
        "batch_size": batch_size,
        "total_steps": 100_000,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "patience": 5_000,
        "num_params": sum(p.numel() for p in model.parameters()),
        "config": {
            "embed_dim": config.embed_dim,
            "mlp_dim": config.mlp_dim,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "context_len": config.context_len,
            "num_layers": config.num_layers,
        }
    }
)

# early stopping
best_loss = float('inf')
patience = 5_000
min_delta = 1e-5  # how small an improvement must be to count

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
total_steps = 100_000
# total_steps = 100
num_warmup_steps = int(0.05 * total_steps)  # 5% warmup
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
running_loss = 0.0

save_path = "/scratch/inath/checkpoints/"
save_interval = 2000  # only check for saving every 500 steps
last_save_step = 0

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

        # Log to wandb
        wandb.log({
            "loss": loss.item(),
            "avg_loss": avg_loss,
            "best_loss": best_loss,
            "learning_rate": lr,
            "gpu_memory_allocated_mb": mem_alloc,
            "gpu_memory_reserved_mb": mem_resv,
            "step": step
        })

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
            if step - last_save_step >= save_interval:
                torch.save(model.state_dict(), save_path + f"tinystories_general_flow_{subset_size or 'full'}_best_model.pt")
                last_save_step = step
        else:
            steps_no_improve += 1
        # if steps_no_improve >= patience:
        #     pbar.write(f"\nEarly stopping at step {step} | best_loss={best_loss:.4f}")
        #     step = total_steps  # force exit
        #     break
pbar.close()
print(f"Training complete at step {step:,} | best_loss={best_loss:.4f}")
torch.save(model.state_dict(), save_path + f"tinystories_general_flow_{subset_size or 'full'}_final_model.pt")
wandb.finish()

# create some examples
with torch.no_grad():
    x0 = torch.full((1, 384), mask_token_id, device=config.device)
    result = gf.forward_sample(x0, dt = 1e-4)
    print(tokenizer.decode(result[0].cpu().numpy()))
# %%w
with torch.no_grad():
    x0 = torch.full((1, 384), mask_token_id, device=config.device)
    result = gf.corrector_sample(x0, dt = 1e-4)
    print(tokenizer.decode(result[0].cpu().numpy()))

# %%
