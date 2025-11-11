"""
Script to analyze saved GAT perplexity evaluation results from pickle files.
Similar structure to evaluate_perplexity.py but adapted for GAT flow matching models.
"""

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import os

# %%

def parse_gat_config_name(name):
    """
    Parse GAT configuration name to extract parameters.
    
    Examples:
    - "no_corrector_temp_0.7" -> {"corrector": False, "temp": 0.7}
    - "corrector_default_temp_1.0" -> {"corrector": True, "temp": 1.0, "alpha": 10, "a": 0.25, "b": 0.25}
    - "corrector_alpha_1_temp_0.8" -> {"corrector": True, "temp": 0.8, "alpha": 1, "a": 0.25, "b": 0.25}
    - "corrector_a_0.5_temp_0.9" -> {"corrector": True, "temp": 0.9, "alpha": 10, "a": 0.5, "b": 0.25}
    """
    config = {
        "corrector": False,
        "temp": None,
        "alpha": 10,  # default
        "a": 0.25,    # default
        "b": 0.25,    # default
    }
    
    # Extract temperature
    if "temp_" in name:
        temp_part = name.split("temp_")[1]
        # Handle cases where there might be more after temp
        temp_str = temp_part.split("_")[0] if "_" in temp_part else temp_part
        try:
            config["temp"] = float(temp_str)
        except ValueError:
            pass
    
    # Check if corrector is used
    if "no_corrector" in name:
        config["corrector"] = False
    elif "corrector" in name:
        config["corrector"] = True
        
        # Extract corrector parameters
        if "alpha_1" in name:
            config["alpha"] = 1
        elif "alpha_" in name:
            # Try to extract alpha value
            parts = name.split("alpha_")
            if len(parts) > 1:
                alpha_str = parts[1].split("_")[0]
                try:
                    config["alpha"] = float(alpha_str)
                except ValueError:
                    pass
        
        if "a_0.5" in name:
            config["a"] = 0.5
        elif "a_" in name:
            # Try to extract a value
            parts = name.split("a_")
            if len(parts) > 1:
                a_str = parts[1].split("_")[0]
                try:
                    config["a"] = float(a_str)
                except ValueError:
                    pass
    
    return config


def analyze_gat_results(pickle_path, output_dir=None):
    """
    Analyze GAT perplexity evaluation results from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file
        output_dir: Optional directory to save plots (if None, displays interactively)
    """
    # Load the results
    print(f"Loading results from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    
    perplexities = results["perplexities"]
    entropies = results["entropies"]
    time_taken = results.get("time_taken", {})
    strings = results.get("strings", {})
    
    print(f"Found {len(perplexities)} configurations")
    print(f"Configurations: {list(perplexities.keys())[:5]}...")
    
    # Parse configurations
    configs = {}
    for name in perplexities.keys():
        configs[name] = parse_gat_config_name(name)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nPerplexity range: {min(perplexities.values()):.2f} - {max(perplexities.values()):.2f}")
    print(f"Entropy range: {min(entropies.values()):.2f} - {max(entropies.values()):.2f}")
    
    if time_taken:
        total_time = sum(time_taken.values())
        print(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        avg_time = np.mean(list(time_taken.values()))
        print(f"Average time per configuration: {avg_time:.2f} seconds")
    
    # Organize data by corrector type and temperature
    data_by_corrector_temp = defaultdict(list)
    no_corrector_data = defaultdict(list)
    
    for name, perplexity in perplexities.items():
        entropy = entropies[name]
        config = configs[name]
        temp = config["temp"]
        
        if temp is None:
            continue
        
        if config["corrector"]:
            # Group by corrector type (default, alpha_1, a_0.5)
            corrector_type = "default"
            if config["alpha"] == 1:
                corrector_type = "alpha_1"
            elif config["a"] == 0.5:
                corrector_type = "a_0.5"
            
            data_by_corrector_temp[(corrector_type, temp)].append({
                "entropy": entropy,
                "perplexity": perplexity,
                "name": name,
                "alpha": config["alpha"],
                "a": config["a"],
                "b": config["b"],
            })
        else:
            no_corrector_data[temp].append({
                "entropy": entropy,
                "perplexity": perplexity,
                "name": name,
            })
    
    # Sort data by entropy for clean plots
    for key in data_by_corrector_temp:
        data_by_corrector_temp[key].sort(key=lambda x: x["entropy"])
    for key in no_corrector_data:
        no_corrector_data[key].sort(key=lambda x: x["entropy"])
    
    # Get unique temperatures and corrector types
    all_temps = sorted(set([config["temp"] for config in configs.values() if config["temp"] is not None]))
    corrector_types = ["default", "alpha_1", "a_0.5"]
    
    # Create color maps
    if len(all_temps) <= 8:
        temp_cmap = plt.cm.Dark2
    elif len(all_temps) <= 12:
        temp_cmap = plt.cm.Set3
    else:
        temp_cmap = plt.cm.tab20
    temp_colors = temp_cmap(np.linspace(0, 1, len(all_temps)))
    temp_to_color = {temp: temp_colors[i] for i, temp in enumerate(all_temps)}
    
    corrector_colors = {"default": "blue", "alpha_1": "green", "a_0.5": "orange"}
    corrector_markers = {"default": "o", "alpha_1": "s", "a_0.5": "^"}
    
    # Plot 1: Perplexity vs Entropy by Temperature (no corrector)
    print("\nGenerating visualizations...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot no_corrector lines
    for temp in sorted(no_corrector_data.keys()):
        data_points = no_corrector_data[temp]
        if not data_points:
            continue
        entropies_vals = [x["entropy"] for x in data_points]
        perplexities_vals = [x["perplexity"] for x in data_points]
        color = temp_to_color[temp]
        plt.plot(entropies_vals, perplexities_vals, color=color, 
                label=f"No Corrector, T={temp}", linewidth=2, linestyle="--", zorder=1)
    
    # Plot corrector lines
    for corrector_type in corrector_types:
        for temp in all_temps:
            key = (corrector_type, temp)
            if key not in data_by_corrector_temp:
                continue
            data_points = data_by_corrector_temp[key]
            if not data_points:
                continue
            entropies_vals = [x["entropy"] for x in data_points]
            perplexities_vals = [x["perplexity"] for x in data_points]
            color = temp_to_color[temp]
            marker = corrector_markers[corrector_type]
            label = f"Corrector {corrector_type}, T={temp}"
            plt.plot(entropies_vals, perplexities_vals, color=color, 
                    marker=marker, label=label, linewidth=2, markersize=8, zorder=2)
    
    plt.xlabel("Entropy", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("GAT Flow: Perplexity vs Entropy by Temperature and Corrector Type", fontsize=14)
    plt.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "gat_perplexity_vs_entropy.png"), dpi=150)
        print(f"Saved: {os.path.join(output_dir, 'gat_perplexity_vs_entropy.png')}")
    else:
        plt.show()
    plt.close()
    
    # Plot 2: Perplexity vs Temperature (grouped by corrector type)
    plt.figure(figsize=(12, 6))
    
    # No corrector
    no_corr_temps = sorted(no_corrector_data.keys())
    no_corr_perps = [no_corrector_data[temp][0]["perplexity"] for temp in no_corr_temps]
    plt.plot(no_corr_temps, no_corr_perps, "o--", label="No Corrector", 
            linewidth=2, markersize=10, color="black")
    
    # Corrector types
    for corrector_type in corrector_types:
        temps = []
        perps = []
        for temp in all_temps:
            key = (corrector_type, temp)
            if key in data_by_corrector_temp and data_by_corrector_temp[key]:
                temps.append(temp)
                perps.append(data_by_corrector_temp[key][0]["perplexity"])
        if temps:
            temps, perps = zip(*sorted(zip(temps, perps)))
            color = corrector_colors[corrector_type]
            marker = corrector_markers[corrector_type]
            plt.plot(temps, perps, f"{marker}-", label=f"Corrector {corrector_type}",
                    linewidth=2, markersize=10, color=color)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("GAT Flow: Perplexity vs Temperature by Corrector Type", fontsize=14)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "gat_perplexity_vs_temperature.png"), dpi=150)
        print(f"Saved: {os.path.join(output_dir, 'gat_perplexity_vs_temperature.png')}")
    else:
        plt.show()
    plt.close()
    
    # Plot 3: Entropy vs Temperature (grouped by corrector type)
    plt.figure(figsize=(12, 6))
    
    # No corrector
    no_corr_entropies = [no_corrector_data[temp][0]["entropy"] for temp in no_corr_temps]
    plt.plot(no_corr_temps, no_corr_entropies, "o--", label="No Corrector",
            linewidth=2, markersize=10, color="black")
    
    # Corrector types
    for corrector_type in corrector_types:
        temps = []
        ents = []
        for temp in all_temps:
            key = (corrector_type, temp)
            if key in data_by_corrector_temp and data_by_corrector_temp[key]:
                temps.append(temp)
                ents.append(data_by_corrector_temp[key][0]["entropy"])
        if temps:
            temps, ents = zip(*sorted(zip(temps, ents)))
            color = corrector_colors[corrector_type]
            marker = corrector_markers[corrector_type]
            plt.plot(temps, ents, f"{marker}-", label=f"Corrector {corrector_type}",
                    linewidth=2, markersize=10, color=color)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.title("GAT Flow: Entropy vs Temperature by Corrector Type", fontsize=14)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "gat_entropy_vs_temperature.png"), dpi=150)
        print(f"Saved: {os.path.join(output_dir, 'gat_entropy_vs_temperature.png')}")
    else:
        plt.show()
    plt.close()
    
    # Print detailed results table
    print("\n=== Detailed Results ===")
    print(f"{'Configuration':<40} {'Temp':<8} {'Perplexity':<12} {'Entropy':<10} {'Time (s)':<10}")
    print("-" * 90)
    
    # Sort by temperature, then by corrector type
    sorted_configs = sorted(perplexities.items(), 
                           key=lambda x: (configs[x[0]]["temp"] or 0, 
                                         "no_corrector" if not configs[x[0]]["corrector"] else "corrector",
                                         x[0]))
    
    for name, perplexity in sorted_configs:
        config = configs[name]
        entropy = entropies[name]
        time_val = time_taken.get(name, 0)
        temp_str = f"{config['temp']:.1f}" if config['temp'] else "N/A"
        print(f"{name:<40} {temp_str:<8} {perplexity:<12.2f} {entropy:<10.2f} {time_val:<10.2f}")
    
    # Print sample generated texts
    if strings:
        print("\n=== Sample Generated Texts ===")
        # Show samples from a few configurations
        sample_configs = list(perplexities.keys())[:3]
        for config_name in sample_configs:
            if config_name in strings and strings[config_name]:
                print(f"\n{config_name}:")
                for i, text in enumerate(strings[config_name][:2]):  # Show first 2 samples
                    print(f"  [{i+1}]: {repr(text[:200])}")
    
    print("\nAnalysis complete!")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GAT perplexity evaluation results")
    parser.add_argument("pickle_path", type=str, nargs="?", 
                       default="/scratch/inath/pickles/tinystories_quadratic_random_flow_perplexity_results.pkl",
                       help="Path to the pickle file containing GAT evaluation results (default: %(default)s)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots (if not provided, plots are displayed)")
    
    args = parser.parse_args()
    
    analyze_gat_results(args.pickle_path, args.output_dir)

