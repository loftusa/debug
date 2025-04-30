# Next Steps for Interpretability Project

Below is a concrete, sequential roadmap that picks up right after Experiment 1 and drives you toward the activation-patching / causal-tracing goals laid out in your Plan.

---

## 1. Choose the "Sweet-Spot" Model & Sequence Length

- **Parse** `experiments/results/**/results_summary.csv` and build a quick table of accuracies.
- **Pick** the smallest model that achieves ≥80% accuracy on at least one sequence length ≥ N (ideally N ≥ 6 for interesting internal computation).
  - Early signs from the plot suggest `Qwen/Qwen2.5-14B-Instruct` ≈ 0.8 at len 2 and ≈ 0.6 at len 5–6 – probably good enough.
- **Freeze** this model/len combo as the "probe target" for interpretability so you stop re-benchmarking.

**Deliverable:**  
`experiments/choose_model.py` → prints & stores (`chosen_model.json`).

---

## 2. Generate Paired Counterfactual Sequences (Step B)

**Goal:** Produce two programs of identical token length where they diverge at an early step k but both sequences are still solved correctly by the chosen model.

- **New util:** `src/debug/paired_sequence_generation.py`
  - `make_counterfactual_pair(seq_len: int, divergence_index: int) -> tuple[Program, Program, dict_metadata]`
  - Ensures same PRNG seed up to k-1, then flips one op/operand at k, keeps rest random but mirrored so lengths match.
- **Script:** `experiments/03_collect_correct_pairs.py`
  - Loop: generate pair → run model on both → keep only if both predictions == ground truth.
  - Store JSONL with fields: `program_a, program_b, divergence_index, ground_truth_a, ground_truth_b, model_logits_a, model_logits_b`.

**Target:** 1-2k valid pairs (enough for averaging heatmaps).

---

## 3. Build Activation Capture & Patching Utils

You need thin, reusable hooks, nothing fancy.

- **Create:** `src/debug/activations.py`
  - Uses `nnsight` (or manual `torch.nn.Module.register_forward_hook`) to record residual stream *and* MLP outputs per layer per token.
  - **API:**
    ```python
    def get_activations(model, tokenizer, prompt: str, tokens_to_keep: slice | list[int]) -> dict[str, torch.Tensor]:
        ...
    ```
- **Create:** `src/debug/patching.py`
  - `patch_and_run(model, tok, prompt_src, prompt_tgt, layer, token_idx, act_key="resid_pre")`
  - Returns ∆logit = P(correct_tgt | patched) - P(correct_tgt | baseline_tgt).

---

## 4. Full Sweep Heatmap (Step C)

**Script:** `experiments/04_activation_heatmap.py`

For each (layer ℓ, token position t) combo:
1. Extract activations from program A (source).
2. Patch into program B at same ℓ, t.
3. Measure ∆logit (or ∆cross-entropy) for the correct final answer token.
4. Average over all stored pairs.

**Save output as:**  
- `heatmap.npy`  
- Matplotlib heatmap like ROME.

---

## 5. Diagnostics & Visualization

- `experiments/plot_heatmap.py` – simple seaborn heatmap.
- `experiments/plot_token_traces.py` – line plot of per-token contribution over layers for one example.

---

## 6. README & Repo Hygiene

- Fill in currently-empty `README.md`:
  - Overview of Experiments 1–4
  - Dataset schema (sequence jsonl, paired jsonl)
  - Chosen model / hyperparams
  - How to reproduce results (`uv run ...`)
  - Folder structure diagram
- Add Ruff pre-commit config (if not present) and short `pytest` for `make_sequence`, `make_counterfactual_pair`.

---

## 7. Stretch Objectives (After Heatmap Works)

- Patch at attention-head granularity (loop over qkv outputs).
- Run causal tracing (replace entire residual streams) to validate variable-value locality.
- Apply DAS/HyperDAS on top of the activation-importance ranking from the heatmap.

---

## Minimal File List to Create

```
src/debug/
    activations.py
    patching.py
    paired_sequence_generation.py
experiments/
    choose_model.py
    03_collect_correct_pairs.py
    04_activation_heatmap.py
    plot_heatmap.py
    plot_token_traces.py
tests/
    test_sequence_generation.py
    test_paired_sequence_generation.py
docs/   (optional visuals)
README.md (updated)
```

> Each script should be a small `click` CLI, follow same pattern as existing code, and respect `uv` runtime.

---

This roadmap keeps code edits minimal while directly advancing toward the interpretability goal. 