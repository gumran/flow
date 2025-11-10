"""
Test script for HybridMaskedSampler.

This script tests the hybrid sampling approach that combines empirical DFM
with a trained Campbell flow model.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

from flow.transformer import IgnorantTransformer
from flow.utils import Config
from flow.campbell_flow import MaskedFMModel
from flow.empirical_dfm import EmpiricalDFM, display_decoded_tokens
from flow.hybrid_flow import HybridMaskedSampler

# %%

# Setup tokenizer and config (matching evaluate_model.py)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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

# Load dataset
dataset_path = "/scratch/inath/datasets/tinystories_10_dataset"
print(f"Loading dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path).with_format("torch")
print(f"Dataset loaded: {len(dataset)} examples")

# Create dataloader for empirical DFM
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
print(f"Created dataloader with batch_size=10")

# %%

# Setup empirical DFM
mask_token_id = tokenizer.mask_token_id
print(f"Mask token ID: {mask_token_id}")

# Create config for empirical DFM (needs num_tokens and context_len)
empirical_config = Config(
    num_tokens=tokenizer.vocab_size,
    context_len=384,
    device=config.device,
    seed=42,
)

empirical_dfm = EmpiricalDFM(
    empirical_config,
    dataloader,
    mask_token_id,
    initial_type="mask",
    tokenizer=tokenizer
)
print("Created EmpiricalDFM with initial_type='mask'")

# %%

# Load trained model
model_path = "/scratch/inath/checkpoints/tinystories_campbell_flow_full_final_model.pt"
print(f"\nLoading trained model from {model_path}...")
model = IgnorantTransformer(config)
model.load_state_dict(torch.load(model_path, map_location=config.device))
model.to(config.device)
model.eval()
print("Model loaded and set to eval mode")

# Wrap model in MaskedFMModel
trained_model = MaskedFMModel(config, model, mask_token_id)
print("Wrapped model in MaskedFMModel")

# %%

# Create hybrid sampler
print("\nCreating HybridMaskedSampler...")
hybrid_sampler = HybridMaskedSampler(config, empirical_dfm, trained_model)
print("HybridMaskedSampler created successfully")

# %%

# Test sampling with different tau values
tau_values = [0.3, 0.5, 0.7]
dt = 0.001
bs = 1

print("\n" + "="*80)
print("Testing hybrid sampling with different tau values")
print("="*80)

for tau in tau_values:
    print(f"\n--- Testing with tau={tau} ---")
    print(f"Sampling with empirical DFM from t=0 to t={tau}, then trained model from t={tau} to t=1...")
    
    try:
        samples = hybrid_sampler.sample(bs=bs, tau=tau, dt=dt)
        print(f"\nSample generated successfully!")
        display_decoded_tokens(samples, tokenizer, f"Hybrid sample (tau={tau})")
    except Exception as e:
        print(f"Error during sampling: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Testing complete!")
print("="*80)

