import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import pipeline configurations
from pipeline_config import BASE_NAME, RESULTS_DIR

def plot_chemotactic_index(ci_csv_path, output_path):
    """
    Generates a distribution plot of the Chemotactic Index across all tracked cells.
    """
    print(f"Loading Chemotactic Index data from {ci_csv_path}...")
    try:
        df = pd.read_csv(ci_csv_path)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    
    sns.histplot(
        data=df, 
        x="Chemotactic_Index", 
        bins=30, 
        kde=True, 
        color="#2b7bba", 
        edgecolor="black",
        alpha=0.7
    )
    
    plt.title("Distribution of Chemotactic Index", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Chemotactic Index (CI)", fontsize=12)
    plt.ylabel("Cell Count", fontsize=12)
    
    mean_ci = df['Chemotactic_Index'].mean()
    plt.axvline(mean_ci, color='red', linestyle='--', linewidth=2, label=f'Mean CI: {mean_ci:.2f}')
    plt.legend()
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_chemotactic_index_by_layer(ci_csv_path, output_path):
    """
    Generates a boxplot of the Chemotactic Index stratified by spatial layer.
    """
    print(f"Loading Layered Chemotactic Index data from {ci_csv_path}...")
    try:
        df = pd.read_csv(ci_csv_path)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    # Filter out anomalous layers (limiting to Layers 1-10 for clean visualization)
    df = df[(df['Starting_Layer'] >= 1) & (df['Starting_Layer'] <= 10)]

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=df, 
        x="Starting_Layer", 
        y="Chemotactic_Index", 
        palette="viridis_r", 
        showfliers=False,    
        width=0.6
    )
    
    plt.title("Chemotactic Index by Starting Spatial Layer", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Initial Spatial Layer (Distance from Wound)", fontsize=12)
    plt.ylabel("Chemotactic Index (CI)", fontsize=12)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_neighbor_velocity_difference(corr_csv_path, output_path):
    """
    Generates a line plot showing the average velocity difference 
    between neighboring cells over time.
    """
    print(f"Loading Neighbor Velocity data from {corr_csv_path}...")
    try:
        df = pd.read_csv(corr_csv_path)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, 
        x="t", 
        y="avg_velocity_vector_distance", 
        marker="o",
        color="#d95f02",
        linewidth=2,
        markersize=8,
        err_style="band"
    )
    
    plt.title("Neighbor Velocity Difference Over Time", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Time (frames)", fontsize=12)
    plt.ylabel("Avg Velocity Difference (µm/min)", fontsize=12) 
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    ci_csv = f"{RESULTS_DIR}/{BASE_NAME}_chemotactic_index.csv"
    corr_csv = f"{RESULTS_DIR}/{BASE_NAME}_neighbor_velocity_difference.csv"
    
    ci_plot_out = f"{RESULTS_DIR}/{BASE_NAME}_plot_chemotactic_index.png"
    ci_layer_plot_out = f"{RESULTS_DIR}/{BASE_NAME}_plot_chemotactic_index_layered.png"
    corr_plot_out = f"{RESULTS_DIR}/{BASE_NAME}_plot_neighbor_correlation.png"
    
    print("\n--- Generating Kinematics Plots ---")
    plot_chemotactic_index(ci_csv, ci_plot_out)
    plot_chemotactic_index_by_layer(ci_csv, ci_layer_plot_out)
    plot_neighbor_velocity_difference(corr_csv, corr_plot_out)
    print("Plotting complete.")