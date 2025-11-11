"""
Script to plot perplexity vs entropy combining results from both Campbell and GAT pickle files.
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

def plot_all_perplexity(campbell_uniform_pickle_path="/scratch/inath/pickles/tinystories_campbell_uniform_flow_evaluation_results.pkl",
                        campbell_flow_pickle_path="/scratch/inath/pickles/tinystories_campbell_flow_evaluation_results.pkl",
                        gat_pickle_path="/scratch/inath/pickles/tinystories_quadratic_random_flow_perplexity_results.pkl",
                        usual_flow_pickle_path="/scratch/inath/pickles/tinystories_usual_flow_perplexity_results.pkl",
                        output_path=None):
    """
    Plot perplexity vs entropy combining results from Campbell, GAT, and UsualFlow pickle files.
    
    Args:
        campbell_uniform_pickle_path: Path to Campbell uniform flow results pickle
        campbell_flow_pickle_path: Path to Campbell flow results pickle
        gat_pickle_path: Path to GAT (Quadratic) results pickle
        usual_flow_pickle_path: Path to UsualFlow results pickle
        output_path: Optional path to save the plot (if None, displays interactively)
    """
    # Load Campbell uniform flow results
    with open(campbell_uniform_pickle_path, "rb") as f:
        campbell_uniform_results = pickle.load(f)

    campbell_uniform_perplexities = campbell_uniform_results["perplexities"]
    campbell_uniform_entropies = campbell_uniform_results["entropies"]
    campbell_uniform_time_taken = campbell_uniform_results.get("time_taken", {})

    # Load Campbell flow results
    with open(campbell_flow_pickle_path, "rb") as f:
        campbell_flow_results = pickle.load(f)

    campbell_flow_perplexities = campbell_flow_results["perplexities"]
    campbell_flow_entropies = campbell_flow_results["entropies"]
    campbell_flow_time_taken = campbell_flow_results.get("time_taken", {})

    # Load GAT results
    with open(gat_pickle_path, "rb") as f:
        gat_results = pickle.load(f)

    gat_perplexities = gat_results["perplexities"]
    gat_entropies = gat_results["entropies"]
    gat_time_taken = gat_results.get("time_taken", {})

    # Load UsualFlow results
    with open(usual_flow_pickle_path, "rb") as f:
        usual_flow_results = pickle.load(f)

    usual_flow_perplexities = usual_flow_results["perplexities"]
    usual_flow_entropies = usual_flow_results["entropies"]
    usual_flow_time_taken = usual_flow_results.get("time_taken", {})
    
    # Print time taken for each model
    print("\n=== Time Taken for Each Model ===")
    
    def print_time_stats(model_name, time_dict):
        if time_dict:
            sorted_configs = sorted(time_dict.items(), key=lambda x: x[0])
            print(f"\n{model_name}:")
            for name, time_val in sorted_configs:
                print(f"  {name}: {time_val:.2f} seconds ({time_val/60:.2f} minutes)")
            total_time = sum(time_dict.values())
            avg_time = np.mean(list(time_dict.values()))
            print(f"  Total: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"  Average: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
        else:
            print(f"\n{model_name}: No time_taken data found")
    
    print_time_stats("Campbell Uniform Flow", campbell_uniform_time_taken)
    print_time_stats("Campbell Flow", campbell_flow_time_taken)
    print_time_stats("GAT (Quadratic)", gat_time_taken)
    print_time_stats("UsualFlow", usual_flow_time_taken)
    
    # Overall statistics
    all_times = []
    all_times.extend(campbell_uniform_time_taken.values())
    all_times.extend(campbell_flow_time_taken.values())
    all_times.extend(gat_time_taken.values())
    all_times.extend(usual_flow_time_taken.values())
    
    if all_times:
        total_all = sum(all_times)
        avg_all = np.mean(all_times)
        print(f"\n=== Overall Statistics ===")
        print(f"Total time across all models: {total_all:.2f} seconds ({total_all/60:.2f} minutes)")
        print(f"Average time per configuration: {avg_all:.2f} seconds ({avg_all/60:.2f} minutes)")
    print()

    # Combine all results
    all_perplexities = {}
    all_entropies = {}

    # Add Campbell uniform flow results with prefix
    for name, ppl in campbell_uniform_perplexities.items():
        all_perplexities[f"CampbellUniform_{name}"] = ppl
        all_entropies[f"CampbellUniform_{name}"] = campbell_uniform_entropies[name]

    # Add Campbell flow results with prefix
    for name, ppl in campbell_flow_perplexities.items():
        all_perplexities[f"CampbellFlow_{name}"] = ppl
        all_entropies[f"CampbellFlow_{name}"] = campbell_flow_entropies[name]

    # Add GAT results with prefix
    for name, ppl in gat_perplexities.items():
        all_perplexities[f"GAT_{name}"] = ppl
        all_entropies[f"GAT_{name}"] = gat_entropies[name]

    # Add UsualFlow results with prefix
    for name, ppl in usual_flow_perplexities.items():
        all_perplexities[f"UsualFlow_{name}"] = ppl
        all_entropies[f"UsualFlow_{name}"] = usual_flow_entropies[name]

    # Organize data by temperature value
    data_by_temp = defaultdict(list)

    # Get all unique eta values for markers
    all_etas = set()

    for name, perplexity in all_perplexities.items():
        entropy = all_entropies[name]
        
        # Handle Campbell uniform flow results
        if name.startswith("CampbellUniform_"):
            campbell_name = name.replace("CampbellUniform_", "")
            
            # Skip top_100 and top_500 entries
            if campbell_name.startswith("top_"):
                continue
            
            # Extract temperature and eta value from name
            temp = None
            eta = 0.0
            
            if "temp_" in campbell_name and "eta_" in campbell_name:
                temp_part = campbell_name.split("temp_")[1].split("_eta_")[0]
                eta_part = campbell_name.split("_eta_")[1]
                temp = float(temp_part)
                eta = float(eta_part)
                all_etas.add(eta)
            elif "eta_" in campbell_name:
                parts = campbell_name.split("_eta_")
                if len(parts) == 2:
                    eta = float(parts[1])
                    all_etas.add(eta)
            all_etas.add(eta)
            
            if temp is not None:
                # Filter: only keep Campbell temperatures 0.7 and 0.8
                if temp in [0.7, 0.8]:
                    data_by_temp[temp].append((entropy, perplexity, name, eta, "CampbellUniform"))
        
        # Handle Campbell flow results
        elif name.startswith("CampbellFlow_"):
            campbell_name = name.replace("CampbellFlow_", "")
            
            # Skip top_100 and top_500 entries
            if campbell_name.startswith("top_"):
                continue
            
            # Extract temperature and eta value from name
            temp = None
            eta = 0.0
            
            if "temp_" in campbell_name and "eta_" in campbell_name:
                temp_part = campbell_name.split("temp_")[1].split("_eta_")[0]
                eta_part = campbell_name.split("_eta_")[1]
                temp = float(temp_part)
                eta = float(eta_part)
                all_etas.add(eta)
            elif "eta_" in campbell_name:
                parts = campbell_name.split("_eta_")
                if len(parts) == 2:
                    eta = float(parts[1])
                    all_etas.add(eta)
            all_etas.add(eta)
            
            if temp is not None:
                # Filter: only keep Campbell temperatures 0.7 and 0.8
                if temp in [0.7, 0.8]:
                    data_by_temp[temp].append((entropy, perplexity, name, eta, "CampbellFlow"))
        
        # Handle GAT results
        elif name.startswith("GAT_"):
            gat_name = name.replace("GAT_", "")
            temp = parse_gat_config_name(gat_name)
            
            if temp is not None:
                # Extract corrector info for labeling
                corrector_type = "no_corrector"
                if "corrector" in gat_name:
                    if "alpha_1" in gat_name:
                        corrector_type = "alpha_1"
                    elif "a_0.5" in gat_name:
                        corrector_type = "a_0.5"
                    else:
                        corrector_type = "default"
                
                # Filter: only keep GAT corrector_alpha_1
                if corrector_type == "alpha_1":
                    # Use eta=0.0 for GAT (we'll distinguish by model type instead)
                    data_by_temp[temp].append((entropy, perplexity, name, 0.0, "GAT", corrector_type))
        
        # Handle UsualFlow results
        elif name.startswith("UsualFlow_"):
            usual_flow_name = name.replace("UsualFlow_", "")
            temp = parse_gat_config_name(usual_flow_name)
            
            if temp is not None:
                # Extract corrector info for labeling
                corrector_type = "no_corrector"
                if "corrector" in usual_flow_name:
                    if "alpha_1" in usual_flow_name:
                        corrector_type = "alpha_1"
                    elif "a_0.5" in usual_flow_name:
                        corrector_type = "a_0.5"
                    else:
                        corrector_type = "default"
                
                # Filter: only keep UsualFlow corrector_alpha_1
                if corrector_type == "alpha_1":
                    data_by_temp[temp].append((entropy, perplexity, name, 0.0, "UsualFlow", corrector_type))

    # Sort each group by entropy for clean line plots
    for temp in data_by_temp:
        data_by_temp[temp].sort(key=lambda x: x[0])

    # Get all unique temperatures and assign colors
    all_temps = sorted(data_by_temp.keys())
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
    plt.figure(figsize=(12, 8))

    # Plot Campbell uniform flow data
    campbell_uniform_temps = set()
    for temp, data_points in sorted(data_by_temp.items()):
        campbell_uniform_points = [p for p in data_points if p[4] == "CampbellUniform"]
        if campbell_uniform_points:
            campbell_uniform_temps.add(temp)
            data_points_sorted = sorted(campbell_uniform_points, key=lambda x: x[0])
            entropies_vals = [x[0] for x in data_points_sorted]
            perplexities_vals = [x[1] for x in data_points_sorted]
            
            label = f"Velocity-Free Uniform T = {temp}"
            color = temp_to_color[temp]
            plt.plot(entropies_vals, perplexities_vals, color=color, 
                    label=label, linewidth=2, linestyle='-', zorder=1)
            
            # Overlay markers for each stochasticity (eta)
            for point in data_points_sorted:
                eta = point[3]
                marker = eta_to_marker.get(eta, 'o')
                plt.scatter(point[0], point[1], marker=marker, color=color,
                           s=60, edgecolors='white', linewidths=0.5, zorder=2)

    # Plot Campbell flow data
    campbell_flow_temps = set()
    for temp, data_points in sorted(data_by_temp.items()):
        campbell_flow_points = [p for p in data_points if p[4] == "CampbellFlow"]
        if campbell_flow_points:
            campbell_flow_temps.add(temp)
            data_points_sorted = sorted(campbell_flow_points, key=lambda x: x[0])
            entropies_vals = [x[0] for x in data_points_sorted]
            perplexities_vals = [x[1] for x in data_points_sorted]
            
            label = f"Velocity-Free Masked T = {temp}"
            color = temp_to_color[temp]
            plt.plot(entropies_vals, perplexities_vals, color=color, 
                    label=label, linewidth=2, linestyle='--', zorder=1)
            
            # Overlay markers for each stochasticity (eta)
            for point in data_points_sorted:
                eta = point[3]
                marker = eta_to_marker.get(eta, 'o')
                plt.scatter(point[0], point[1], marker=marker, color=color,
                           s=60, edgecolors='white', linewidths=0.5, zorder=2)

    # Plot GAT data - collect all alpha_1 points across all temperatures
    gat_alpha1_points = []
    for temp, data_points in sorted(data_by_temp.items()):
        gat_points = [p for p in data_points if p[4] == "GAT"]
        for point in gat_points:
            corrector_type = point[5] if len(point) > 5 else "no_corrector"
            if corrector_type == "alpha_1":
                gat_alpha1_points.append(point)
    
    # Plot GAT as a single line connecting all points
    if gat_alpha1_points:
        # Sort by entropy for clean line
        gat_alpha1_points_sorted = sorted(gat_alpha1_points, key=lambda x: x[0])
        entropies_vals = [x[0] for x in gat_alpha1_points_sorted]
        perplexities_vals = [x[1] for x in gat_alpha1_points_sorted]
        
        plt.plot(entropies_vals, perplexities_vals, color='red', 
                label='Masked Noisy Interpolation with Correction', linewidth=2, linestyle='--', 
                marker='P', markersize=10, zorder=3)

    # Plot UsualFlow data - collect all alpha_1 points across all temperatures
    usual_flow_alpha1_points = []
    for temp, data_points in sorted(data_by_temp.items()):
        usual_flow_points = [p for p in data_points if p[4] == "UsualFlow"]
        for point in usual_flow_points:
            corrector_type = point[5] if len(point) > 5 else "no_corrector"
            if corrector_type == "alpha_1":
                usual_flow_alpha1_points.append(point)
    
    # Plot UsualFlow as a single line connecting all points
    if usual_flow_alpha1_points:
        # Sort by entropy for clean line
        usual_flow_alpha1_points_sorted = sorted(usual_flow_alpha1_points, key=lambda x: x[0])
        entropies_vals = [x[0] for x in usual_flow_alpha1_points_sorted]
        perplexities_vals = [x[1] for x in usual_flow_alpha1_points_sorted]
        
        plt.plot(entropies_vals, perplexities_vals, color='blue', 
                label='Masked with Corrector', linewidth=2, linestyle=':', 
                marker='+', markersize=10, zorder=3)

    plt.xlabel("Entropy", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Perplexity vs Entropy by Model", fontsize=14)

    # Create main legend for model lines (on the left)
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
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

# %%

# Run the function
plot_all_perplexity(output_path="/scratch/inath/figures/all_perplexity_vs_entropy.png")

# %%

