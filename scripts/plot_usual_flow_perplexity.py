"""
Script to plot perplexity vs entropy for UsualFlow results only.
"""

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# %%

def parse_gat_config_name(name):
    """Parse GAT configuration name to extract temperature."""
    temp = None
    if "temp_" in name:
        temp_part = name.split("temp_")[1]
        temp_str = temp_part.split("_")[0] if "_" in temp_part else temp_part
        try:
            temp = float(temp_str)
        except ValueError:
            pass
    return temp

# %%

def plot_usual_flow_perplexity(usual_flow_pickle_path="/scratch/inath/pickles/tinystories_usual_flow_perplexity_results.pkl",
                               output_path=None):
    """
    Plot perplexity vs entropy for all UsualFlow results.
    
    Args:
        usual_flow_pickle_path: Path to UsualFlow results pickle
        output_path: Optional path to save the plot (if None, displays interactively)
    """
    # Load UsualFlow results
    with open(usual_flow_pickle_path, "rb") as f:
        usual_flow_results = pickle.load(f)

    usual_flow_perplexities = usual_flow_results["perplexities"]
    usual_flow_entropies = usual_flow_results["entropies"]
    usual_flow_time_taken = usual_flow_results.get("time_taken", {})
    
    # Print time taken for each configuration
    print("\n=== Time Taken for Each Configuration ===")
    if usual_flow_time_taken:
        # Sort by configuration name
        sorted_configs = sorted(usual_flow_time_taken.items(), key=lambda x: x[0])
        for name, time_val in sorted_configs:
            print(f"{name}: {time_val:.2f} seconds ({time_val/60:.2f} minutes)")
        
        total_time = sum(usual_flow_time_taken.values())
        avg_time = np.mean(list(usual_flow_time_taken.values()))
        print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average time per configuration: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
    else:
        print("No time_taken data found in pickle file.")
    print()

    # Organize data by temperature and corrector type
    data_by_temp_corrector = defaultdict(list)

    for name, perplexity in usual_flow_perplexities.items():
        entropy = usual_flow_entropies[name]
        temp = parse_gat_config_name(name)
        
        if temp is not None:
            # Extract corrector info for labeling
            corrector_type = "no_corrector"
            if "corrector" in name:
                if "alpha_1" in name:
                    corrector_type = "alpha_1"
                elif "a_0.5" in name:
                    corrector_type = "a_0.5"
                else:
                    corrector_type = "default"
            
            # Include all configurations
            data_by_temp_corrector[(temp, corrector_type)].append((entropy, perplexity, name))

    # Sort each group by entropy for clean line plots
    for key in data_by_temp_corrector:
        data_by_temp_corrector[key].sort(key=lambda x: x[0])

    # Get all unique temperatures and corrector types
    all_temps = sorted(set([k[0] for k in data_by_temp_corrector.keys()]))
    all_corrector_types = sorted(set([k[1] for k in data_by_temp_corrector.keys()]))

    # Assign colors to temperatures
    if len(all_temps) <= 8:
        cmap = plt.cm.Dark2
    elif len(all_temps) <= 12:
        cmap = plt.cm.Set3
    else:
        cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(all_temps)))
    temp_to_color = {temp: colors[i] for i, temp in enumerate(all_temps)}

    # Assign linestyles and markers to corrector types
    corrector_styles = {
        "no_corrector": ("-", "o"),
        "alpha_1": ("--", "s"),
        "a_0.5": ("-.", "^"),
        "default": (":", "D")
    }

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot all configurations grouped by temperature and corrector type
    for temp in all_temps:
        for corrector_type in all_corrector_types:
            key = (temp, corrector_type)
            if key not in data_by_temp_corrector:
                continue
            
            data_points = data_by_temp_corrector[key]
            if not data_points:
                continue
            
            entropies_vals = [x[0] for x in data_points]
            perplexities_vals = [x[1] for x in data_points]
            
            color = temp_to_color[temp]
            linestyle, marker = corrector_styles.get(corrector_type, ("-", "o"))
            
            # Format corrector type for legend
            if corrector_type == "a_0.5":
                corrector_label = "a=0.5"
            elif corrector_type == "alpha_1":
                corrector_label = "alpha=1"
            else:
                corrector_label = corrector_type
            
            label = f"T = {temp}, {corrector_label}"
            plt.scatter(entropies_vals, perplexities_vals, color=color, 
                       label=label, marker=marker, s=100, zorder=2)

    plt.xlabel("Entropy", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Perplexity vs Entropy by Model (Masked with Corrector)", fontsize=14)
    plt.ylim(0, 200)

    # Create legend
    plt.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

# %%

# Run the function
plot_usual_flow_perplexity(output_path="/scratch/inath/figures/usual_flow_perplexity_vs_entropy.png")

# %%

