#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main() -> None:
    base = Path(__file__).parent / "results"
    all_dfs = []
    for seq_dir in sorted(base.glob("seq_len_*")):
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
        return

    full = pd.concat(all_dfs, ignore_index=True)
    pivot = full.pivot(index="seq_len", columns="model_id", values="accuracy")
    pivot.sort_index(inplace=True)

    ax = pivot.plot(marker="o", figsize=(12, 8))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs. Sequence Length")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("accuracy_vs_sequence_length.png")
    plt.show()

if __name__ == "__main__":
    main()