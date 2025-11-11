"""
Experiment script to evaluate hybrid sampler across different tau values.

This script tests the hybrid sampler with tau values from 0.1 to 0.9, generates samples,
and measures their distance to the training dataset using sentence embeddings.
"""

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from flow.transformer import IgnorantTransformer
from flow.utils import Config
from flow.campbell_flow import MaskedFMModel
from flow.empirical_dfm import EmpiricalDFM
from flow.hybrid_flow import HybridMaskedSampler
from flow.eval import SentenceBERTEvaluator

# %%
# Setup configuration
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

config = Config(
    num_tokens=len(tokenizer),
    embed_dim=512,
    mlp_dim=2048,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=64,
    context_len=32,
    num_layers=16,
    timestep_scale=1000.0,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# %%
# Load dataset
dataset_path = "/scratch/agumran/datasets/32_tok_tinystories_8192_dataset"
print(f"Loading dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path).with_format("torch")
print(f"Dataset loaded: {len(dataset)} examples")

# Create dataloader for empirical DFM
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
print(f"Created dataloader with batch_size=10")

# %%
# Setup evaluator and cache embeddings
print("\nSetting up SentenceBERTEvaluator...")
evaluator = SentenceBERTEvaluator(device=config.device)

print("Decoding entire training dataset to text...")
training_texts = []
for i in tqdm(range(len(dataset)), desc="Decoding dataset"):
    tokens = dataset[i]['input_ids']
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    training_texts.append(text)

print(f"Decoded {len(training_texts)} training examples")

print("Caching embeddings for training dataset...")
evaluator.cache_embeddings(training_texts)
print("Embeddings cached successfully")

# %%
# Setup empirical DFM
mask_token_id = tokenizer.mask_token_id
print(f"\nMask token ID: {mask_token_id}")

# Create config for empirical DFM
empirical_config = Config(
    num_tokens=tokenizer.vocab_size,
    context_len=32,
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
model_path = "/scratch/agumran/checkpoints/tinystories_campbell_masked_flow_8192_final_model.pt"
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
# Run experiment loop
tau_values = [0.2, 0.4, 0.6, 0.8]
dt = 0.001
bs = 128

print("\n" + "="*80)
print("Running tau sweep experiment")
print("="*80)

tau_distances = []
tau_means = []
tau_stds = []

for tau in tqdm(tau_values, desc="Testing tau values"):
    print(f"\n--- Testing with tau={tau} ---")
    
    # Generate samples
    print(f"Generating {bs} samples...")
    with torch.inference_mode():
        samples = hybrid_sampler.sample(bs=bs, tau=tau, dt=dt)
    
    # Decode samples to text
    generated_texts = []
    for i in range(bs):
        text = tokenizer.decode(samples[i].cpu().numpy(), skip_special_tokens=True)
        generated_texts.append(text)
    
    # Calculate distances
    print("Calculating distances to training dataset...")
    distances, closest_indices = evaluator.calculate_distance_batch(generated_texts)
    distances = distances.cpu().numpy()
    
    mean_distance = distances.mean()
    std_distance = distances.std()
    
    tau_distances.append(distances)
    tau_means.append(mean_distance)
    tau_stds.append(std_distance)
    
    print(f"Mean distance: {mean_distance:.4f} ± {std_distance:.4f}")
    print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

print("\n" + "="*80)
print("Experiment complete!")
print("="*80)

# %%
# Visualization
print("\nGenerating plots...")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Plot tau vs distance
plt.figure(figsize=(10, 6))
plt.errorbar(tau_values, tau_means, yerr=tau_stds, 
             marker='o', linestyle='-', capsize=5, capthick=2, markersize=8)
plt.xlabel('Tau (transition point)', fontsize=12)
plt.ylabel('Mean Distance to Training Dataset', fontsize=12)
plt.title('Hybrid Sampler: Distance vs Tau', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
figure_path = "figures/tau_vs_distance.png"
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {figure_path}")
plt.close()

# Save raw data
data_path = "results/tau_distance_data.pt"
torch.save({
    'tau_values': tau_values,
    'tau_distances': tau_distances,  # List of numpy arrays
    'tau_means': tau_means,
    'tau_stds': tau_stds,
}, data_path)
print(f"Saved raw data to {data_path}")

print("\nAll done!")

