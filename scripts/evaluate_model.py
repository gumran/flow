"""
Script to evaluate the generalisability properties of the flow models.
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt


from flow.transformer import TimeAwareTransformer, IgnorantTransformer
from flow.utils import Config
from flow.campbell_flow import MaskedFMModel, UniformFMModel
from flow.general_flow import UsualFlow
from flow.eval import test_perplexity_evaluator

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
    # output_channels=2,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)
config = large_config

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

def analyze_logits(logits, token_id):
    logits_values = logits[0, token_id, 0, :].detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(logits_values, bins=50)
    plt.title("Histogram of Logits")
    plt.xlabel("Logit Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.show()

    topk = 20
    logits_vector = logits[0, token_id, 0, :]
    top_logits, top_indices = torch.topk(logits_vector, topk)
    print("Top 20 logits and their token ids:")
    for rank, (score, idx) in enumerate(zip(top_logits, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        print(f"{rank+1:2d}: token id {idx.item():5d}, logit {score.item():9.4f}, as text: '{token_str}'")


    num_samples = len(ds)
    print(num_samples)
    for i in range(num_samples):  # Look at all samples
        token = ds[i]["input_ids"][token_id]
        print(f"Sample {i}: token id {token}, as text: '{tokenizer.decode([token]) if token is not None else 'None'}'")

# %%

dataset_path = "/scratch/inath/datasets/tinystories_10_dataset"
ds = load_from_disk(dataset_path)
print(ds)
# %%

print(ds['input_ids'][0])
print(tokenizer.decode(ds['input_ids'][0]))

# %%

model_path = "/scratch/inath/checkpoints/tinystories_campbell_flow_full_final_model.pt"
model = IgnorantTransformer(config)
model.load_state_dict(torch.load(model_path))
model.to(config.device)
model.eval()
fm = MaskedFMModel(config, model)

def analyze_logits(logits, token_id):
    logits_values = logits[0, token_id, 0, :].detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(logits_values, bins=50)
    plt.title("Histogram of Logits")
    plt.xlabel("Logit Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.show()

    topk = 20
    logits_vector = logits[0, token_id, 0, :]
    top_logits, top_indices = torch.topk(logits_vector, topk)
    print("Top 20 logits and their token ids:")
    for rank, (score, idx) in enumerate(zip(top_logits, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        print(f"{rank+1:2d}: token id {idx.item():5d}, logit {score.item():9.4f}, as text: '{token_str}'")


    num_samples = len(ds)
    print(num_samples)
    for i in range(num_samples):  # Look at all samples
        token = ds[i]["input_ids"][token_id]
        print(f"Sample {i}: token id {token}, as text: '{tokenizer.decode([token]) if token is not None else 'None'}'")

# %%

x1 = torch.tensor(ds[[0]]["input_ids"], device=config.device)
# %%
logits = model(x1, torch.tensor([0.0], device=config.device))
analyze_logits(logits, 10)
# %%

t = torch.tensor([0.01], device=config.device)
rand_mask = (torch.rand_like(x1.float(), device=config.device) < t)
xt = torch.where(rand_mask, x1, tokenizer.mask_token_id)
print(xt)
# %%
logits = model(xt, t)
analyze_logits(logits, 10)

# %%
