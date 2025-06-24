#%%
import sys
sys.path.append('../src')

from debug import ExperimentRunner, generators, prompts, quick_experiment

def main():
    MODELS = [
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    ]
    variable_binding_config = quick_experiment(
        name="variable_binding",
        prompt_template=prompts.VARIABLE_BINDING,
        program_generator=generators.make_variable_binding_program,
        models=MODELS,
        num_seqs=100,
        seq_lens=list(range(1, 18))
    )

    runner = ExperimentRunner()
    result = runner.run(variable_binding_config, no_cache=True)
    
    # plots/analysis
    runner.plot(variable_binding_config.name)
    runner.analyze_errors(variable_binding_config.name, n_examples=5)

    print("Experiment completed!")
    print(f"Overall accuracy: {result['summary']['overall_accuracy']:.1%}")
    print(f"Results saved to: {result['output_dir']}")

if __name__ == "__main__":
    main()


#%%
# import numpy as np
# MODELS = [
# "Qwen/Qwen3-0.6B",
# "Qwen/Qwen3-1.7B",
# "Qwen/Qwen3-4B",
# "Qwen/Qwen3-8B",
# "Qwen/Qwen3-14B",
# ]
# variable_binding_config = quick_experiment(
#     name="variable_binding",
#     prompt_template=prompts.VARIABLE_BINDING,
#     program_generator=generators.make_variable_binding_program,
#     models=MODELS,
#     num_seqs=100,
#     seq_lens=list(range(1, 100))
# )

# runner = ExperimentRunner()

# seq_len = 60
# master_rng = np.random.RandomState(12345)
# program, true_answer = variable_binding_config.program_generator(seq_len, master_rng)
# prompt = variable_binding_config.prompt_template.format(code=program)

# print(prompt)



#%%
# import numpy as np
# from debug import ExperimentRunner, generators, prompts, quick_experiment

# MODELS = [
# # "Qwen/Qwen3-0.6B",
# # "Qwen/Qwen3-1.7B",
# # "Qwen/Qwen3-4B",
# "Qwen/Qwen3-8B",
# "Qwen/Qwen3-14B",
# ]
# variable_binding_config = quick_experiment(
#     name="variable_binding",
#     prompt_template=prompts.VARIABLE_BINDING,
#     program_generator=generators.make_variable_binding_program,
#     models=MODELS,
#     num_seqs=100,
#     seq_lens=list(range(1, 100))
# )

# runner = ExperimentRunner()
# config = variable_binding_config
# seq_len = 5
# master_rng = np.random.RandomState(12345)
# program, true_answer = config.program_generator(seq_len, master_rng)
# prompt = config.prompt_template.format(code=program)
# print(prompt)






#%%

# %%

#%%
# # Load existing results and plot them
# import json
# import pandas as pd
# import matplotlib.pyplot as plt

# def load_and_plot_existing_results():
#     """Load existing variable_binding results and create plots."""
#     results_path = "results/debug_experiments/variable_binding_20250528_142250/results.json"
    
#     print("Loading existing results...")
#     with open(results_path, 'r') as f:
#         results = json.load(f)
    
#     print(f"Loaded {len(results)} results")
    
#     # Convert to DataFrame for easier plotting
#     df = pd.DataFrame(results)
    
#     # Create the plot
#     plt.figure(figsize=(12, 8))
    
#     # Plot accuracy by sequence length for each model
#     for model in df["model_id"].unique():
#         model_data = df[df["model_id"] == model]
#         seq_lens = []
#         accuracies = []
        
#         for seq_len in sorted(model_data["seq_len"].unique()):
#             seq_data = model_data[model_data["seq_len"] == seq_len]
#             accuracy = seq_data["correct"].mean()
#             seq_lens.append(seq_len)
#             accuracies.append(accuracy)
        
#         # Clean up model name for legend
#         model_name = model.split('/')[-1]
#         plt.plot(seq_lens, accuracies, marker='o', label=model_name, linewidth=2, markersize=6)
    
#     plt.xlabel("Sequence Length", fontsize=12)
#     plt.ylabel("Accuracy", fontsize=12) 
#     plt.title("Variable Binding: Accuracy vs Sequence Length", fontsize=14, fontweight='bold')
#     plt.legend(fontsize=10)
#     plt.grid(True, alpha=0.3)
#     plt.ylim(0, 1)
    
#     # Add some styling
#     plt.tight_layout()
    
#     # Save the plot
#     plt.savefig("results/debug_experiments/variable_binding_20250528_142250/accuracy_plot.png", 
#                 dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Print summary statistics
#     print("\n=== Summary Statistics ===")
#     print(f"Overall accuracy: {df['correct'].mean():.1%}")
#     print("\nBy model:")
#     for model, acc in df.groupby("model_id")["correct"].mean().items():
#         model_name = model.split('/')[-1]
#         print(f"  {model_name}: {acc:.1%}")
    
#     print("\nBy sequence length:")
#     for seq_len, acc in df.groupby("seq_len")["correct"].mean().items():
#         print(f"  Length {seq_len}: {acc:.1%}")

# # Run the analysis
# load_and_plot_existing_results()

# # %%