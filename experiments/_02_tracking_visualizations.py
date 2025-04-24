#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

base = Path(__file__).parent / "results"
all_dfs = []
for seq_dir in sorted(base.glob("seq_len_*")):
    print(seq_dir)
    csv_path = seq_dir / "results_summary.csv"
    if not csv_path.exists():
        continue
    # parse seq_len from directory name
    try:
        seq_len = int(seq_dir.name.split("_")[2])
    except (IndexError, ValueError):
        continue
    df = pd.read_csv(csv_path)
    if "seq_len" not in df.columns:
        df["seq_len"] = seq_len
    else:
        df["seq_len"] = df["seq_len"].astype(int)
    all_dfs.append(df)

if not all_dfs:
    print("No result CSVs found under 'results/seq_len_*'")
    exit()

#%%
full = pd.concat(all_dfs, ignore_index=True)
pivot = full.pivot(index="seq_len", columns="model_id", values="accuracy")
pivot.sort_index(inplace=True)

pivot = pivot.loc[:, pivot.mean(axis=0) > 0.45]
pivot = pivot.reindex(columns=pivot.mean().sort_values(ascending=False).index)
pivot
#%%
# Calculate average accuracy across all models for each sequence length
pivot['average'] = pivot.mean(axis=1)
#%%
# Set style and color palette
sns.set_style("whitegrid")
colors = sns.color_palette("viridis", n_colors=len(pivot.columns)-1)  # -1 for the average column
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']
linestyles = ['-', '--', '-.', ':']

plt.figure(figsize=(14, 10))

# Plot each model with a unique color, marker, and line style
for i, column in enumerate(pivot[['openai/gpt-4o']]):  # Skip the last column (average)
    marker = markers[i % len(markers)]
    linestyle = linestyles[i % len(linestyles)]
    plt.plot(
        pivot.index, 
        pivot[column], 
        marker=marker, 
        linestyle=linestyle,
        linewidth=2.0,
        markersize=8,
        label=column,
        color=colors[i],
        alpha=0.7
    )

# Plot the average with emphasis
plt.plot(
    pivot.index,
    pivot['average'],
    marker='o',
    linestyle='-',
    linewidth=4.0,
    markersize=10,
    label='Average',
    color='red',
    zorder=10  # Put average line on top
)

# Add labels and title
plt.xlabel("Sequence Length", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Model Accuracy vs. Sequence Length", fontsize=16, fontweight='bold')

# Customize grid
plt.grid(True, linestyle='--', alpha=0.7)

# Improve x-axis ticks
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
    title_fontsize=13
)

# Show max values on plot
for column in pivot.columns[:-1]:  # Skip average for individual annotations
    max_idx = pivot[column].idxmax()
    max_val = pivot[column].max()
    plt.annotate(
        f"{max_val:.3f}",
        (max_idx, max_val),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        fontweight='bold'
    )

# Annotate average points
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
plt.savefig("accuracy_vs_sequence_length_gpt4o.png", dpi=300, bbox_inches="tight")
plt.show()