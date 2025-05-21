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
    help="Directory containing the summary_integer_all.csv file.",
)
@click.option(
    "--model_ids",
    default=None,
    help="Comma-separated list of model IDs to include in the plot. If not provided, all models will be included.",
    type=click.types.STRING, # Keep as string, will split later
) # Changed to accept comma-separated string for consistency with other scripts
def main(input_dir_str: str, model_ids: Optional[str]): # model_ids is now Optional[str]
    """Generates visualizations from integer arithmetic experiment results.""" # Updated docstring
    input_dir = Path(input_dir_str)
    summary_csv_path = input_dir / "summary_integer_all.csv" # New summary file name

    print(f"Looking for results file: {summary_csv_path}")

    if not summary_csv_path.exists():
        print(f"  {summary_csv_path.name} not found in {input_dir}. Exiting.")
        return

    print(f"  Loading {summary_csv_path}")
    try:
        full_df = pd.read_csv(summary_csv_path)
    except Exception as e:
        print(f"Error loading CSV {summary_csv_path}: {e}")
        return

    if full_df.empty:
        print(f"No data found in {summary_csv_path}. Exiting.")
        return

    # Ensure required columns exist
    required_cols = ["seq_len", "model_id", "accuracy_avg_pair"]
    if not all(col in full_df.columns for col in required_cols):
        print(f"CSV {summary_csv_path} is missing one or more required columns: {required_cols}. Found: {full_df.columns.tolist()}")
        return

    # Use "accuracy_avg_pair" as the value for pivoting
    try:
        pivot = full_df.pivot(index="seq_len", columns="model_id", values="accuracy_avg_pair")
        pivot.sort_index(inplace=True)
    except Exception as e:
        print(f"Error pivoting data: {e}. Check CSV format and content.")
        return


    # Handle model_ids if provided as a comma-separated string
    selected_model_ids_list: Optional[List[str]] = None
    if model_ids:
        selected_model_ids_list = [m.strip() for m in model_ids.split(',')]
        # Filter pivot table for selected models
        # Ensure that the columns actually exist in the pivot table to avoid KeyErrors
        existing_models_in_pivot = [m for m in selected_model_ids_list if m in pivot.columns]
        missing_models = [m for m in selected_model_ids_list if m not in pivot.columns]
        if missing_models:
            print(f"Warning: The following specified model_ids were not found in the data: {missing_models}")
        if not existing_models_in_pivot:
            print(f"Warning: None of the specified model_ids ({selected_model_ids_list}) were found. Plotting all available models.")
        else:
            pivot = pivot[existing_models_in_pivot]


    # Optional: Filter models with low average accuracy (adjust threshold as needed)
    # pivot = pivot.loc[:, pivot.mean(axis=0) > 0.45] # Example filter
    if not pivot.empty:
        pivot = pivot.reindex(columns=pivot.mean().sort_values(ascending=False).index)
    
    print("\nPivot table of accuracies (using accuracy_avg_pair):")
    print(pivot)

    if pivot.empty:
        print("Pivot table is empty (possibly due to filtering or no data for selected models). Cannot generate plot.")
        return
        
    # Calculate average accuracy across all (selected or all) models for each sequence length
    pivot["average"] = pivot.mean(axis=1)
    
    sns.set_style("whitegrid")
    num_models = len([col for col in pivot.columns if col != 'average']) # Exclude average
    colors = sns.color_palette("viridis", n_colors=max(num_models, 1))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "X", "d"]
    linestyles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(14, 10))

    plotted_columns = [col for col in pivot.columns if col != 'average']
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
            alpha=0.7,
        )

    if 'average' in pivot.columns and not pivot['average'].dropna().empty:
        plt.plot(
            pivot.index,
            pivot["average"],
            marker="o",
            linestyle="-",
            linewidth=4.0,
            markersize=10,
            label="Average",
            color="red",
            zorder=10,
        )

    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Average Accuracy (Pair)", fontsize=14) # Updated Y-axis label
    title = "Model Accuracy (Integer Arithmetic) vs. Sequence Length" # Updated title
    plt.title(title, fontsize=16, fontweight="bold")

    plt.grid(True, linestyle="--", alpha=0.7)

    if not pivot.index.empty:
        # Ensure x-ticks are integers if sequence lengths are integers
        if pd.api.types.is_integer_dtype(pivot.index):
            plt.xticks(ticks=pivot.index, labels=[str(int(x)) for x in pivot.index], fontsize=12)
        else:
            plt.xticks(pivot.index, fontsize=12)
            
    plt.yticks(fontsize=12)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=12,
        title="Models",
        title_fontsize=13,
    )

    if 'average' in pivot.columns and not pivot['average'].dropna().empty:
        for idx in pivot.index:
            if pd.notna(pivot.loc[idx, 'average']):
                avg_val = pivot.loc[idx, 'average']
                plt.annotate(
                    f"{avg_val:.3f}",
                    (idx, avg_val),
                    xytext=(0, -15),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight='bold',
                    color='red',
                    ha='center'
                )
        
        # Check if there's any non-NA average value before finding max
        if pivot['average'].notna().any():
            max_avg_idx = pivot['average'].idxmax()
            max_avg_val = pivot['average'].max()
            plt.axhline(y=max_avg_val, color='red', linestyle='--', alpha=0.4)
            if not pivot.index.empty:
                plt.annotate(
                    f"Max Avg: {max_avg_val:.3f} at seq_len={max_avg_idx}",
                    (pivot.index[0], max_avg_val), # Position annotation relative to plot
                    xytext=(10, 5),
                    textcoords="offset points",
                    fontsize=12,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
                )


    plt.tight_layout()
    # Output filename now reflects the new task and structure
    output_filename = input_dir / "accuracy_vs_sequence_length_integer.png"
    print(f"Saving plot to {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    # plt.show() # Optionally show plot, or rely on saved file.

if __name__ == "__main__":
    main()