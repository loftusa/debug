#%%
"""
Visualize full token/layer patching results.

This script loads the data from the most recent patching experiment
and provides a simple example for creating a heatmap visualization.
It is intended to be used interactively in a notebook.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer
import click

# Add src to PYTHONPATH for custom module imports
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parents[1] / "src"
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from debug.causal_visualization import plot_causal_flow_heatmap
from debug.causal_tracing import InterventionResult

#/share/u/lofty/code_llm/debug/results/full_token_layer_patching/20250615_185732/intervention_results.json
def create_visualization(model_name, base_results_dir=None):
    """Create causal flow heatmap visualization for a specific model."""
    # --- Configuration ---------------------------------------------------------
    # Find the latest experiment directory automatically
    if base_results_dir is None:
        base_results_dir = Path(__file__).resolve().parents[1] / "results" / "full_token_layer_patching_1_hop_seq_5"
    else:
        base_results_dir = Path(base_results_dir)
        
    if not base_results_dir.exists():
        raise FileNotFoundError(f"Base results directory not found: {base_results_dir}")

    # Get the most recent timestamped experiment folder
    try:
        latest_experiment_dir = sorted(
            [d for d in base_results_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True
        )[0]
    except IndexError:
        latest_experiment_dir = base_results_dir

    assert latest_experiment_dir.exists(), f"Latest experiment directory not found: {latest_experiment_dir}"

    # Find results file for the specified model
    MODEL = model_name
    RESULTS_FILE = latest_experiment_dir / "intervention_results.json"
    if not RESULTS_FILE.exists():
        RESULTS_FILE = latest_experiment_dir / "experiment_results.json"
    if not RESULTS_FILE.exists():
        RESULTS_FILE = latest_experiment_dir / f"{MODEL}/intervention_results.json"
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found in: {latest_experiment_dir}")

    print(f"Loading results from: {RESULTS_FILE}")
    
    # Load the results into a pandas DataFrame for easy inspection
    results_df = load_results_as_dataframe(RESULTS_FILE)
    print("Results loaded successfully. DataFrame preview:")
    print(results_df.head())
    
    # Convert DataFrame to InterventionResult objects for visualization
    intervention_results = convert_to_intervention_results(results_df)

    # Get token labels from the program text
    if not results_df.empty and "original_program" in results_df.columns and pd.notna(results_df["original_program"].iloc[0]):
        program_text = results_df["original_program"].iloc[0]
        # For real analysis, you would get the tokenizer name from the experiment config
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        token_labels = clean_token_labels(tokenizer.tokenize(program_text))
        print(f"\nGenerated {len(token_labels)} token labels from saved program text.")
    else:
        # Fallback for older result files without the program text
        print("\n`original_program` not found in results, using placeholder token labels.")
        num_tokens = results_df["token_pos"].max() + 1
        token_labels = [f"tok_{i}" for i in range(num_tokens)]

    # Create the heatmap of logit differences
    print("\nGenerating heatmap...")
    fig, ax = plot_causal_flow_heatmap(
        intervention_results=intervention_results,
        token_labels=token_labels,
    )

    MODEL_TITLE = MODEL or ''
    ax.set_title(f"Logit Difference Heatmap {MODEL_TITLE}")

    # Save the plot to the same directory as the results file
    output_dir = Path(RESULTS_FILE).parent
    plot_filename = f"causal_flow_heatmap_{MODEL_TITLE.replace('/', '_')}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {plot_path}")
    plt.close()  # Close instead of show for non-interactive use
    
    return str(plot_path)


def load_results_as_dataframe(path: Path) -> pd.DataFrame:
    """Load patching results into a pandas DataFrame."""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle both old list format and new dictionary format
    if isinstance(data, dict) and "intervention_results" in data:
        return pd.DataFrame(data["intervention_results"])
    else:
        return pd.DataFrame(data)

def convert_to_intervention_results(df: pd.DataFrame) -> list[InterventionResult]:
    """Convert a DataFrame row into an InterventionResult object for visualization funcs."""
    results = []
    for _, row in df.iterrows():
        results.append(
            InterventionResult(
                intervention_type=row.get("intervention_type", "residual_stream"),
                layer_idx=row.get("layer_idx"),
                head_idx=row.get("head_idx"),
                target_token_pos=row.get("target_token_pos") or row.get("token_pos"),
                logit_difference=row.get("logit_difference"),
                normalized_logit_difference=row.get("normalized_logit_difference"),
                success_rate=row.get("success_rate"),
                original_program=row.get("original_program"),
                counterfactual_program=row.get("counterfactual_program"),
                token_labels=row.get("token_labels"),
                target_name=row.get("target_name"),
            )
        )
    return results

def clean_token_labels(token_labels):
    """Clean up tokenizer artifacts for better visualization."""
    if not token_labels:
        return token_labels
    
    clean_labels = []
    for token in token_labels:
        # Handle Unicode escape sequences that appear in JSON
        if isinstance(token, str):
            # Replace common tokenizer prefixes
            clean_token = token.replace('\u0120', ' ')  # Ġ space prefix
            clean_token = clean_token.replace('\u010a', '\\n')  # newline
            clean_token = clean_token.replace('Ġ', ' ')  # Direct Ġ character
            clean_token = clean_token.replace('▁', ' ')  # SentencePiece space
            
            # Handle other common prefixes
            if clean_token.startswith('##'):
                clean_token = clean_token[2:]
                
            clean_labels.append(clean_token)
        else:
            clean_labels.append(str(token))
    
    return clean_labels

@click.command()
@click.option("--model", default="Qwen_Qwen3-8B", help="Model name to generate visualization for")
@click.option("--base-results-dir", default=None, help="Base results directory path")
def main(model, base_results_dir):
    """Generate causal flow heatmap visualization for a specific model."""
    figure_path = create_visualization(model, base_results_dir)
    print(f"Visualization complete: {figure_path}")


if __name__ == "__main__":
    main()
