"""
Script to analyze entropy evolution of predicted distributions in flow matching models.
Compares empirical DFM and trained Campbell flow model (masked variant).
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from flow.transformer import IgnorantTransformer
from flow.utils import Config
from flow.campbell_flow import MaskedFMModel
from flow.empirical_dfm import EmpiricalDFM

# %%
# Setup configuration (same as evaluate_model.py)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

large_config = Config(
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
config = large_config

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# %%
def calculate_entropy(probabilities):
    """
    Calculate entropy from probability distributions.
    
    Args:
        probabilities: Tensor of shape (bs, context_len, num_tokens) or (context_len, num_tokens)
    
    Returns:
        entropy: Tensor of shape (bs, context_len) or (context_len,)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    probs = probabilities + eps
    
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log(probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return entropy

# %%
# Load trained model
print("Loading trained model...")
model_path = "/scratch/agumran/checkpoints/tinystories_campbell_masked_flow_1024_final_model.pt"
model = IgnorantTransformer(config)
model.load_state_dict(torch.load(model_path, map_location=config.device))
model.to(config.device)
model.eval()
mask_token_id = tokenizer.mask_token_id
trained_fm = MaskedFMModel(config, model, mask_token_id=mask_token_id)

# %%
# Load dataset for EmpiricalDFM
print("Loading dataset for EmpiricalDFM...")
dataset_path = "/scratch/agumran/datasets/32_tok_tinystories_1024_dataset"
dataset = load_from_disk(dataset_path).with_format("torch")
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
empirical_dfm = EmpiricalDFM(config, dataloader, mask_token_id=mask_token_id, 
                              initial_type="mask", tokenizer=tokenizer)

# %%
# Get test examples
print("Preparing test examples...")
num_test_examples = 1024
test_indices = list(range(num_test_examples))
test_tokens = torch.stack([dataset[i]['input_ids'].clone() for i in test_indices])
test_tokens = test_tokens.to(config.device)

print(f"Test examples shape: {test_tokens.shape}")
for i in range(num_test_examples):
    print(f"Example {i}: {tokenizer.decode(test_tokens[i].cpu().numpy()[:50])}...")

# %%
# Analyze entropy evolution over time
print("Analyzing entropy evolution...")
timesteps = np.linspace(0, 1, 101)  # [0, 0.1, 0.2, ..., 1.0]
bs = test_tokens.shape[0]
c = config.context_len

# Storage for results
trained_entropies = []  # List of (bs, c) tensors, one per timestep
empirical_entropies = []  # List of (bs, c) tensors, one per timestep
masking_rates = []  # List of masking rates per timestep

with torch.no_grad():
    for t_val in tqdm(timesteps, desc="Processing timesteps"):
        t_tensor = torch.full((bs,), t_val, device=config.device)
        
        # Generate noisy samples at this timestep
        xt_trained = trained_fm.sample_t(test_tokens, t_tensor)
        
        # Identify masked positions (only calculate entropy for these)
        masked_positions = (xt_trained == mask_token_id)  # (bs, c)
        masking_rate = masked_positions.float().mean().item()
        masking_rates.append(masking_rate)
        
        # Get probabilities from trained model
        logits_trained = model(xt_trained, t_tensor)  # (bs, c, num_tokens)
        # Ensure logits are in correct shape (bs, c, num_tokens)
        if logits_trained.dim() == 4:
            # If output_channels > 1, take first channel
            logits_trained = logits_trained[:, :, 0, :]
        probs_trained = F.softmax(logits_trained, dim=-1)
        entropy_trained = calculate_entropy(probs_trained)  # (bs, c)
        
        # Set entropy to NaN for non-masked positions
        entropy_trained = entropy_trained.cpu().numpy()
        entropy_trained[~masked_positions.cpu().numpy()] = np.nan
        trained_entropies.append(entropy_trained)
        
        # Get probabilities from empirical DFM
        # Note: calculate_masked_probabilities doesn't use t, but accepts it
        probs_empirical = empirical_dfm.calculate_masked_probabilities(xt_trained, t_tensor)  # (bs, c, num_tokens)
        entropy_empirical = calculate_entropy(probs_empirical)  # (bs, c)
        
        # Set entropy to NaN for non-masked positions
        entropy_empirical = entropy_empirical.cpu().numpy()
        entropy_empirical[~masked_positions.cpu().numpy()] = np.nan
        empirical_entropies.append(entropy_empirical)

# Convert to numpy for plotting (already numpy arrays)
trained_entropies = np.stack(trained_entropies)  # (num_timesteps, bs, c)
empirical_entropies = np.stack(empirical_entropies)  # (num_timesteps, bs, c)

# %%
# Compute statistics
print("\n=== Summary Statistics ===")

# Print masking rate statistics
print(f"\nMasking rate statistics:")
print(f"  At t=0: {masking_rates[0]:.4f} ({masking_rates[0]*100:.2f}% masked)")
print(f"  At t=0.5: {masking_rates[len(masking_rates)//2]:.4f} ({masking_rates[len(masking_rates)//2]*100:.2f}% masked)")
print(f"  At t=1: {masking_rates[-1]:.4f} ({masking_rates[-1]*100:.2f}% masked)")

# Average entropy over masked positions only (using nanmean to ignore NaN values)
trained_mean_entropy = np.nanmean(trained_entropies, axis=(1, 2))  # (num_timesteps,)
empirical_mean_entropy = np.nanmean(empirical_entropies, axis=(1, 2))  # (num_timesteps,)

print(f"\nInitial entropy (t=0):")
print(f"  Trained model: {trained_mean_entropy[0]:.4f}")
print(f"  Empirical DFM: {empirical_mean_entropy[0]:.4f}")

print(f"\nFinal entropy (t=1):")
print(f"  Trained model: {trained_mean_entropy[-1]:.4f}")
print(f"  Empirical DFM: {empirical_mean_entropy[-1]:.4f}")

print(f"\nEntropy decrease:")
print(f"  Trained model: {trained_mean_entropy[0] - trained_mean_entropy[-1]:.4f}")
print(f"  Empirical DFM: {empirical_mean_entropy[0] - empirical_mean_entropy[-1]:.4f}")

# Rate of entropy decrease (average slope)
trained_slope = np.mean(np.diff(trained_mean_entropy))
empirical_slope = np.mean(np.diff(empirical_mean_entropy))
print(f"\nAverage entropy decrease rate (per 0.1 timestep):")
print(f"  Trained model: {trained_slope:.4f}")
print(f"  Empirical DFM: {empirical_slope:.4f}")

# %%
# Visualization
print("\nGenerating visualizations...")

# Plot 1: Average entropy vs time
plt.figure(figsize=(10, 6))
plt.plot(timesteps, trained_mean_entropy, 'o-', label='Trained Model', linewidth=2, markersize=6)
plt.plot(timesteps, empirical_mean_entropy, 's-', label='Empirical DFM', linewidth=2, markersize=6)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Average Entropy', fontsize=12)
plt.title('Average Entropy Evolution Over Time (Masked Tokens Only)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/entropy_vs_time_average.png', dpi=150)
print("Saved: figures/entropy_vs_time_average.png")
plt.close()

# Plot 2: Entropy vs time for specific token positions
selected_positions = [0, 50, 100, 200]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, pos in enumerate(selected_positions):
    if pos >= c:
        continue
    ax = axes[idx]
    
    # Average over examples (using nanmean to handle NaN values)
    trained_pos_entropy = np.nanmean(trained_entropies[:, :, pos], axis=1)
    empirical_pos_entropy = np.nanmean(empirical_entropies[:, :, pos], axis=1)
    
    ax.plot(timesteps, trained_pos_entropy, 'o-', label='Trained Model', linewidth=2, markersize=5)
    ax.plot(timesteps, empirical_pos_entropy, 's-', label='Empirical DFM', linewidth=2, markersize=5)
    ax.set_xlabel('Time (t)', fontsize=10)
    ax.set_ylabel('Entropy', fontsize=10)
    ax.set_title(f'Position {pos}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Entropy Evolution at Specific Token Positions (Masked Tokens Only)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/entropy_vs_time_positions.png', dpi=150)
print("Saved: figures/entropy_vs_time_positions.png")
plt.close()

# Plot 3: Heatmap of entropy over time and position
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Average over examples for heatmap (using nanmean to handle NaN values)
trained_heatmap = np.nanmean(trained_entropies, axis=1).T  # (c, num_timesteps)
empirical_heatmap = np.nanmean(empirical_entropies, axis=1).T  # (c, num_timesteps)

# Trained model heatmap
im1 = axes[0].imshow(trained_heatmap, aspect='auto', cmap='viridis', 
                     extent=[timesteps[0], timesteps[-1], c-1, 0])
axes[0].set_xlabel('Time (t)', fontsize=12)
axes[0].set_ylabel('Token Position', fontsize=12)
axes[0].set_title('Trained Model Entropy Heatmap (Masked Tokens Only)', fontsize=13)
plt.colorbar(im1, ax=axes[0], label='Entropy')

# Empirical DFM heatmap
im2 = axes[1].imshow(empirical_heatmap, aspect='auto', cmap='viridis',
                     extent=[timesteps[0], timesteps[-1], c-1, 0])
axes[1].set_xlabel('Time (t)', fontsize=12)
axes[1].set_ylabel('Token Position', fontsize=12)
axes[1].set_title('Empirical DFM Entropy Heatmap (Masked Tokens Only)', fontsize=13)
plt.colorbar(im2, ax=axes[1], label='Entropy')

plt.tight_layout()
plt.savefig('figures/entropy_heatmap.png', dpi=150)
print("Saved: figures/entropy_heatmap.png")
plt.close()

# Plot 4: Entropy difference (trained - empirical)
entropy_diff = trained_mean_entropy - empirical_mean_entropy
plt.figure(figsize=(10, 6))
plt.plot(timesteps, entropy_diff, 'o-', color='purple', linewidth=2, markersize=6)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Entropy Difference (Trained - Empirical)', fontsize=12)
plt.title('Difference in Average Entropy Between Models (Masked Tokens Only)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/entropy_difference.png', dpi=150)
print("Saved: figures/entropy_difference.png")
plt.close()

print("\nAnalysis complete! All plots saved.")

