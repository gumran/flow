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

from flow.transformer import TimeAwareTransformer, Config
from flow.campbell_flow import MaskedFMModel, UniformFMModel
from flow.general_flow import UsualMaskedFlow

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
    num_layers=17,
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

dataset_path = "/scratch/inath/datasets/tinystories_1_dataset"
ds = load_from_disk(dataset_path)
print(ds)
# %%

print(ds['input_ids'][0])
print(tokenizer.decode(ds['input_ids'][0]))

# %%

model_path = "/scratch/inath/checkpoints/tinystories_general_flow_1_model.pt"
model = TimeAwareTransformer(config)
model.load_state_dict(torch.load(model_path))
model.to(config.device)
model.eval()
gf = UsualMaskedFlow(config, model)
# %%

with torch.no_grad():
    x0 = torch.full((1, 384), tokenizer.mask_token_id, device=config.device)
    output = gf.forward_sample(x0)
    print(tokenizer.decode(output[0]))
# %%
