#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import click
from typing import Optional, List

@click.command()
@click.option(
    "--kind",
    default="groups",
    type=click.Choice(["groups", "ops"]),
    help="Kind of results to visualize (determines input filename).",
)
# either None or a list of model IDs
@click.option(
    "--model_ids",
    default=None,
    help="List of model IDs to include in the plot. If not provided, all models will be included.",
    type=click.types.STRING,
    multiple=True,
)
def main(kind: str, model_ids: Optional[List[str]]):
    """Generates visualizations from sequence length experiment results."""
    base = Path(__file__).parent / "results"
    all_dfs = []
    result_filename = (
        "results_summary.csv" if kind == "groups" else "results_summary_ops.csv"
    )
    print(f"Looking for results file: {result_filename}")

    # Determine the correct directory pattern based on kind
    if kind == "ops":
        glob_pattern = "seq_len_*_ops"
    else:  # kind == "groups"
        # For groups, initially glob all seq_len_* and filter later
        glob_pattern = "seq_len_*"

    for seq_dir in sorted(base.glob(glob_pattern)):
        # If kind is "groups", explicitly skip directories ending with "_ops"
        if kind == "groups" and seq_dir.name.endswith("_ops"):
            print(f"  Skipping ops-specific directory {seq_dir.name} for groups visualization.")
            continue

        print(f"Checking directory: {seq_dir}")
        csv_path = seq_dir / result_filename
        if not csv_path.exists():
            print(f"  {result_filename} not found, skipping.")
            continue
        # parse seq_len from directory name
        try:
            seq_len = int(seq_dir.name.split("_")[2])
        except (IndexError, ValueError):
            print(f"  Could not parse seq_len from {seq_dir.name}, skipping.")
            continue

        print(f"  Loading {csv_path}")
        df = pd.read_csv(csv_path)
        if "seq_len" not in df.columns:
            df["seq_len"] = seq_len
        else:
            df["seq_len"] = df["seq_len"].astype(int)
        all_dfs.append(df)

    if not all_dfs:
        print(f"No result CSVs ('{result_filename}') found under 'results/seq_len_*'")
        return  # Use return instead of exit in a function

    # %%
    full = pd.concat(all_dfs, ignore_index=True)
    pivot = full.pivot(index="seq_len", columns="model_id", values="accuracy")
    pivot.sort_index(inplace=True)
    if model_ids is not None:
        assert isinstance(model_ids, list), "model_ids must be a list"
        pivot = pivot.loc[:, model_ids]

    # Optional: Filter models with low average accuracy (adjust threshold as needed)
    # pivot = pivot.loc[:, pivot.mean(axis=0) > 0.45]
    pivot = pivot.reindex(columns=pivot.mean().sort_values(ascending=False).index)
    print("\nPivot table of accuracies:")
    print(pivot)
    # %%
    # Calculate average accuracy across all models for each sequence length
    pivot["average"] = pivot.mean(axis=1)
    # %%
    # Set style and color palette
    sns.set_style("whitegrid")
    # Ensure enough colors if many models exist
    num_models = len(pivot.columns) - 1 # Exclude average
    colors = sns.color_palette("viridis", n_colors=max(num_models, 1)) # Handle case with 0 models
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "X", "d"]
    linestyles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(14, 10))

    # Plot each model with a unique color, marker, and line style
    plotted_columns = [col for col in pivot.columns if col != 'average'] # Explicitly exclude 'average'
    for i, column in enumerate(plotted_columns):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        color_index = i % len(colors) # Ensure color index is within bounds
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

    # Plot the average with emphasis if there are models to average
    if 'average' in pivot.columns:
        plt.plot(
            pivot.index,
            pivot["average"],
            marker="o",
            linestyle="-",
            linewidth=4.0,
            markersize=10,
            label="Average",
            color="red",
            zorder=10,  # Put average line on top
        )

    # Add labels and title
    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    title = f"Model Accuracy vs. Sequence Length ({kind.capitalize()} Results)"
    plt.title(title, fontsize=16, fontweight="bold")


    # Customize grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Improve x-axis ticks
    if not pivot.index.empty:
        plt.xticks(pivot.index, fontsize=12)
    plt.yticks(fontsize=12)


    # Add legend with better placement and formatting
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=12,
        title="Models",
        title_fontsize=13,
    )

    # # Show max values on plot - Removed for clarity, can be added back if needed
    # for column in plotted_columns:
    #     if not pivot[column].empty:
    #         max_idx = pivot[column].idxmax()
    #         max_val = pivot[column].max()
    #         plt.annotate(
    #             f"{max_val:.3f}",
    #             (max_idx, max_val),
    #             xytext=(5, 5),
    #             textcoords="offset points",
    #             fontsize=9,
    #             fontweight='bold'
    #         )

    # Annotate average points if average exists
    if 'average' in pivot.columns and not pivot['average'].empty:
        for idx in pivot.index:
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

        # Add a horizontal line at maximum average accuracy
        max_avg_idx = pivot['average'].idxmax()
        max_avg_val = pivot['average'].max()
        plt.axhline(y=max_avg_val, color='red', linestyle='--', alpha=0.4)
        # Ensure pivot.index is not empty before accessing its first element
        if not pivot.index.empty:
            plt.annotate(
                f"Max Avg: {max_avg_val:.3f} at seq_len={max_avg_idx}",
                (pivot.index[0], max_avg_val),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight='bold',
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
            )


    plt.tight_layout()
    output_filename = f"accuracy_vs_sequence_length_{kind}.png" # Use kind in filename
    print(f"Saving plot to {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()