#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import click
from typing import Optional, List

@click.command()
@click.option(
    "--input-dir",
    "input_dir_str",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing the summary CSV file for boolean experiments."
)
@click.option(
    "--model-ids",
    "model_ids_filter", # Renamed to avoid conflict with pandas column name
    default=None,
    help="Comma-separated list of model IDs to include in the plot. If not provided, all models in the CSV will be included.",
    type=str,
)
def main(input_dir_str: str, model_ids_filter: Optional[str]):
    """Generates visualizations from boolean experiment results."""
    input_dir = Path(input_dir_str)
    summary_csv_path = input_dir / "summary_boolean_all.csv"

    if not summary_csv_path.exists():
        print(f"Error: Summary CSV file not found at {summary_csv_path}")
        return

    print(f"Loading summary data from: {summary_csv_path}")
    full_df = pd.read_csv(summary_csv_path)

    if full_df.empty:
        print(f"No data found in {summary_csv_path}")
        return

    # Ensure correct data types
    full_df["seq_len"] = full_df["seq_len"].astype(int)
    full_df["accuracy_avg_pair"] = pd.to_numeric(full_df["accuracy_avg_pair"], errors='coerce')
    full_df.dropna(subset=["accuracy_avg_pair"], inplace=True) # Drop rows where conversion failed

    # Filter by model_ids if provided
    if model_ids_filter:
        selected_models = [m.strip() for m in model_ids_filter.split(',')]
        full_df = full_df[full_df["model_id"].isin(selected_models)]
        if full_df.empty:
            print(f"No data found for the specified model IDs: {selected_models}")
            return

    pivot = full_df.pivot(index="seq_len", columns="model_id", values="accuracy_avg_pair")
    pivot.sort_index(inplace=True)
    
    # Reorder columns by mean accuracy (descending)
    if not pivot.empty:
        pivot = pivot.reindex(columns=pivot.mean().sort_values(ascending=False).index)

    print("\nPivot table of average pair accuracies:")
    print(pivot)

    # Calculate average accuracy across all plotted models for each sequence length
    if not pivot.empty:
        pivot["Average (Plotted Models)"] = pivot.mean(axis=1)
    
    # Set style and color palette
    sns.set_style("whitegrid")
    num_models_to_plot = len(pivot.columns) - (1 if "Average (Plotted Models)" in pivot.columns else 0)
    colors = sns.color_palette("viridis", n_colors=max(num_models_to_plot, 1))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "X", "d"]
    linestyles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(16, 10)) # Increased figure size for better legend placement

    plotted_columns = [col for col in pivot.columns if col != 'Average (Plotted Models)']
    for i, column in enumerate(plotted_columns):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        color_index = i % len(colors)
        plt.plot(
            pivot.index,
            pivot[column],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=8,
            label=column,
            color=colors[color_index],
            alpha=0.8,
        )

    if 'Average (Plotted Models)' in pivot.columns and not pivot['Average (Plotted Models)'].empty:
        plt.plot(
            pivot.index,
            pivot["Average (Plotted Models)"],
            marker="o",
            linestyle="-",
            linewidth=3.5,
            markersize=10,
            label="Average (Plotted Models)",
            color="dodgerblue", # Changed average color for better contrast
            zorder=10, 
        )

    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Average Pair Accuracy (0 or 1)", fontsize=14)
    title = "Model Average Pair Accuracy vs. Sequence Length (Boolean Task)"
    plt.title(title, fontsize=16, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)

    if not pivot.index.empty:
        plt.xticks(sorted(list(set(pivot.index.tolist()))), fontsize=12) # Ensure all unique seq_lens are ticks
    plt.yticks(fontsize=12)

    plt.legend(
        bbox_to_anchor=(1.02, 1), # Adjust anchor for better placement with larger figure
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=11, # Slightly smaller font for legend if many models
        title="Models",
        title_fontsize=12,
    )

    if 'Average (Plotted Models)' in pivot.columns and not pivot['Average (Plotted Models)'].empty:
        for idx in pivot.index:
            avg_val = pivot.loc[idx, 'Average (Plotted Models)']
            if pd.notna(avg_val):
                plt.annotate(
                    f"{avg_val:.3f}",
                    (idx, avg_val),
                    xytext=(0, -15 if len(plotted_columns) > 5 else 10), # Adjust based on number of lines
                    textcoords="offset points",
                    fontsize=9,
                    fontweight='bold',
                    color='dodgerblue',
                    ha='center'
                )
        max_avg_val = pivot['Average (Plotted Models)'].max()
        if pd.notna(max_avg_val) and not pivot.index.empty:
            max_avg_idx = pivot['Average (Plotted Models)'].idxmax()
            plt.axhline(y=max_avg_val, color='dodgerblue', linestyle=':', alpha=0.5)
            plt.annotate(
                f"Max Avg: {max_avg_val:.3f} at seq_len={max_avg_idx}",
                (pivot.index.min(), max_avg_val),
                xytext=(10, 7),
                textcoords="offset points",
                fontsize=11,
                fontweight='bold',
                color='dodgerblue',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="dodgerblue", alpha=0.8)
            )

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend on the right
    output_filename = input_dir / "accuracy_vs_sequence_length_boolean.png"
    print(f"Saving plot to {output_filename}")
    plt.savefig(output_filename, dpi=300)
    # plt.show() # Typically not needed for automated scripts

if __name__ == "__main__":
    main() 