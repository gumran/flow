"""
Script to evaluate the perplexity of pre-trained flow matching models
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
import time
from collections import defaultdict
import pickle


from flow.transformer import TimeAwareTransformer, IgnorantTransformer
from flow.utils import Config
from flow.campbell_flow import MaskedFMModel, UniformFMModel
from flow.general_flow import UsualFlow
from flow.eval import PerplexityEvaluator

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def evaluate_entropy(text_samples, tokenizer):
    """
    text_samples: List[str], iterables of raw text
    tokenizer: Huggingface tokenizer or similar
    Returns: average entropy per token position, computed over tokenized samples
    """
    # Tokenize samples with tokenizer
    encodings = tokenizer(
        text_samples,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = encodings["input_ids"]  # shape: (num_samples, seq_len)
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
    # input_ids is (num_samples, seq_len)
    num_samples, seq_len = input_ids.shape
    entropies = []

    for pos in range(seq_len):
        tokens_at_pos = input_ids[:, pos]  # tensor of shape (num_samples,)
        # Count occurrences for each token id
        token_ids, counts = torch.unique(tokens_at_pos, return_counts=True)
        probs = counts.float() / counts.sum()
        # Compute entropy at this position (add epsilon for numerical stability)
        entropy_pos = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        entropies.append(entropy_pos)

    avg_entropy = float(torch.tensor(entropies).mean())
    return avg_entropy

small_config = Config(
    num_tokens=len(tokenizer),
    embed_dim=128,
    mlp_dim=256,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=16,
    context_len=32,
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

# %%

path = "/scratch/inath/checkpoints/tinystories_campbell_uniform_flow_full_final_model.pt"
model = IgnorantTransformer(large_config)
model.load_state_dict(torch.load(path))
model.to(config.device)
model.eval()
fm = UniformFMModel(config, model)

# %%

perplexity_evaluator = PerplexityEvaluator(device =  config.device)


# %%

# with torch.inference_mode():
#     generated_sentences = fm.sample(10, dt = 0.01, temperature = 1.0)
#     generated_texts = tokenizer.batch_decode(generated_sentences, skip_special_tokens=True)

# print(generated_texts[0])
# %%

args_dict = {}

# Add top_k configs (top_100 and top_500)
args_dict["top_100"] = {
    "dt": 0.01,
    "temperature": 1.0,
    "eta": 0.0,
    "top_k": 100,
}
args_dict["top_500"] = {
    "dt": 0.01,
    "temperature": 1.0,
    "eta": 0.0,
    "top_k": 500,
}

# Add all combinations of temperature and eta (without top_k, i.e., unmasked)
temperatures = [0.7, 0.8, 0.9, 1.0]
etas = [0.0, 3.0, 15.0]

for temp in temperatures:
    for eta in etas:
        name = f"temp_{temp}_eta_{eta}"
        # Only include 'eta' if eta != 0.0 (default)
        args = {
            "dt": 0.01,
            "temperature": temp,
        }
        if eta != 0.0:
            args["eta"] = eta
        args_dict[name] = args


time_taken = defaultdict(float)
strings = defaultdict(list)
perplexities = defaultdict(float)
entropies = defaultdict(float)

def evaluate_samples(name, args):
    start_time = time.time()
    with torch.inference_mode():
        while len(strings[name]) < 100:
            generated_sentences = fm.sample(10, **args)
            generated_texts = tokenizer.batch_decode(generated_sentences, skip_special_tokens=True)
            strings[name].extend(generated_texts)
    end_time = time.time()
    time_taken[name] = end_time - start_time
    perplexities[name] = perplexity_evaluator.calculate_perplexity(strings[name])
    entropies[name] = evaluate_entropy(strings[name], tokenizer)

for name, args in args_dict.items():
    evaluate_samples(name, args)

# %%


with open("/scratch/inath/pickles/tinystories_campbell_uniform_flow_evaluation_results.pkl", "wb") as f:
    pickle.dump({
        "time_taken": dict(time_taken),
        "strings": dict(strings),
        "perplexities": dict(perplexities),
        "entropies": dict(entropies)
    }, f)

# %%

# Test reading the pickle file and print its contents to verify the pickle
with open("/scratch/inath/pickles/tinystories_campbell_uniform_flow_evaluation_results.pkl", "rb") as f:
    loaded_results = pickle.load(f)
    print("Loaded pickle contents:")
    for k, v in loaded_results.items():
        print(f"{k}: {type(v)}")
        # Print a small portion of content for inspection
        if isinstance(v, dict):
            for subk, subv in list(v.items())[:2]:
                print(f"  {subk}: {str(subv)[:200]} ...")
        else:
            print(f"Value: {str(v)[:200]} ...")

# %%
import numpy as np

# Load the results
with open("/scratch/inath/pickles/tinystories_campbell_uniform_flow_evaluation_results.pkl", "rb") as f:
    results = pickle.load(f)

perplexities = results["perplexities"]
entropies = results["entropies"]

# Organize data by temperature value
# Extract temperature and eta from model names like "temp_0.7_eta_3.0" or "top_100"
data_by_temp = defaultdict(list)

# Store top_k entries separately for individual dots
top_k_points = {}

# Get all unique eta values for markers
all_etas = set()

for name, perplexity in perplexities.items():
    # Collect top_100 and top_500 entries separately
    if name.startswith("top_"):
        entropy = entropies[name]
        top_k_points[name] = (entropy, perplexity)
        continue
    
    entropy = entropies[name]
    
    # Extract temperature and eta value from name
    temp = None
    eta = 0.0
    
    if "temp_" in name and "eta_" in name:
        # Extract temperature and eta (e.g., "temp_0.7_eta_3.0" -> temp=0.7, eta=3.0)
        temp_part = name.split("temp_")[1].split("_eta_")[0]
        eta_part = name.split("_eta_")[1]
        temp = float(temp_part)
        eta = float(eta_part)
        all_etas.add(eta)
    elif "eta_" in name:
        # Extract eta value only
        parts = name.split("_eta_")
        if len(parts) == 2:
            eta = float(parts[1])
            all_etas.add(eta)
    # For entries without eta, treat as eta=0.0
    all_etas.add(eta)
    
    if temp is not None:
        data_by_temp[temp].append((entropy, perplexity, name, eta))

# Sort each group by entropy for clean line plots
for temp in data_by_temp:
    data_by_temp[temp].sort(key=lambda x: x[0])

# Get all unique temperatures and assign colors
all_temps = sorted(data_by_temp.keys())
# Use Dark2 colormap for better color distinction (designed for categorical data)
# Alternative options: Set2, Set3, tab20, or plasma/viridis for sequential
if len(all_temps) <= 8:
    cmap = plt.cm.Dark2
elif len(all_temps) <= 12:
    cmap = plt.cm.Set3
else:
    cmap = plt.cm.tab20
colors = cmap(np.linspace(0, 1, len(all_temps)))
temp_to_color = {temp: colors[i] for i, temp in enumerate(all_temps)}

# Get all unique eta values and assign markers
all_etas = sorted(all_etas)
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']
eta_to_marker = {eta: markers[i % len(markers)] for i, eta in enumerate(all_etas)}

# Create the plot
plt.figure(figsize=(10, 6))

for temp, data_points in sorted(data_by_temp.items()):
    # Sort all points by entropy for this temperature
    data_points_sorted = sorted(data_points, key=lambda x: x[0])
    entropies_vals = [x[0] for x in data_points_sorted]
    perplexities_vals = [x[1] for x in data_points_sorted]
    
    # Plot the line connecting all points for this temperature
    label = f"T = {temp}"
    color = temp_to_color[temp]
    plt.plot(entropies_vals, perplexities_vals, color=color, 
             label=label, linewidth=2, zorder=1)
    
    # Overlay markers for each stochasticity (eta)
    for point in data_points_sorted:
        eta = point[3]  # point[3] is eta
        marker = eta_to_marker.get(eta, 'o')
        plt.scatter(point[0], point[1], marker=marker, color=color,
                   s=60, edgecolors='white', linewidths=0.5, zorder=2)

# Plot top_k entries as individual dots
for name, (entropy, perplexity) in top_k_points.items():
    plt.scatter(entropy, perplexity, s=100, marker='s', 
                label=name, edgecolors='black', linewidths=1.5, zorder=5)

plt.xlabel("Entropy", fontsize=12)
plt.ylabel("Perplexity", fontsize=12)
plt.title("Perplexity vs Entropy by Temperature", fontsize=14)

# Create main legend for temperature lines and top_k points (on the left)
legend1 = plt.legend(fontsize=10, loc='upper left', framealpha=0.9)

# Add second legend for stochasticity (eta) markers (on the right)
eta_legend_elements = []
for eta in all_etas:
    marker = eta_to_marker[eta]
    eta_legend_elements.append(plt.Line2D([0], [0], marker=marker, 
                                         color='gray', linestyle='None',
                                         markersize=8, label=f'η = {eta}'))
if eta_legend_elements:
    legend2 = plt.legend(handles=eta_legend_elements, fontsize=9, 
                        loc='upper right', title='Stochasticity (η)', framealpha=0.9)
    plt.gca().add_artist(legend1)  # Re-add the first legend

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

