#%%
import sys
sys.path.append('../src')

from debug import ExperimentRunner, generators, prompts, quick_experiment

MODELS = [
# "Qwen/Qwen3-0.6B",
# "Qwen/Qwen3-1.7B",
# "Qwen/Qwen3-4B",
"Qwen/Qwen3-8B",
# "Qwen/Qwen3-14B",
]
variable_binding_config = quick_experiment(
    name="variable_binding",
    prompt_template=prompts.VARIABLE_BINDING,
    program_generator=generators.make_variable_binding_program,
    models=MODELS,
    num_seqs=100,
    seq_lens=list(range(1, 100))
)

runner = ExperimentRunner()
#%%
import numpy as np
seq_len = 17 
master_rng = np.random.RandomState(12345)
program, true_answer = variable_binding_config.program_generator(seq_len, master_rng)
prompt = variable_binding_config.prompt_template.format(code=program)
print(prompt)
#%%

# questions
# - 


# result = runner.run(variable_binding_config, no_cache=True)

# plots/analysis
# runner.plot(variable_binding_config.name)
# runner.analyze_errors(variable_binding_config.name, n_examples=5)

# print("Experiment completed!")
# print(f"Overall accuracy: {result['summary']['overall_accuracy']:.1%}")
# print(f"Results saved to: {result['output_dir']}")

#%%