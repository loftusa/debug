import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_ROOT = Path(__file__).parent / "results"


def load_results() -> pd.DataFrame:
    """Load all results_summary.csv files under results/len_* sub‑dirs.

    Returns:
        pd.DataFrame: Concatenated data with columns [model_id, seq_len, accuracy].
    """
    records: List[Dict[str, str]] = []
    for sub in sorted(RESULTS_ROOT.glob("len_*/results_summary.csv")):
        seq_len_str = sub.parent.name.split("len_")[-1]
        try:
            seq_len = int(seq_len_str)
        except ValueError:
            continue
        with sub.open() as f:
            reader = csv.DictReader(f, fieldnames=["model_id", "seq_len", "accuracy"])
            # skip header row if present
            for row in reader:
                if row["model_id"] == "model_id":
                    continue
                records.append({
                    "model_id": row["model_id"],
                    "seq_len": int(row.get("seq_len", seq_len)),
                    "accuracy": float(row["accuracy"]),
                })
    if not records:
        raise FileNotFoundError("No results found under 'results/len_*/results_summary.csv'.")
    return pd.DataFrame(records)


def plot(df: pd.DataFrame) -> None:
    """Plot accuracy vs sequence length for each model and save the figure."""
    plt.figure(figsize=(10, 6))
    for model_id, grp in df.groupby("model_id"):
        grp_sorted = grp.sort_values("seq_len")
        plt.plot(grp_sorted["seq_len"], grp_sorted["accuracy"], marker="o", label=model_id)
    plt.xlabel("Sequence length")
    plt.ylabel("Accuracy")
    plt.title("Model accuracy vs. sequence length")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    output_path = RESULTS_ROOT / "sequence_length_accuracy.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    df_all = load_results()
    plot(df_all) 