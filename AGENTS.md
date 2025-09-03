# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/debug/` (core framework: `runner.py`, `core.py`, generators, prompts, causal tracing and visualization).
- Experiments: `experiments/` (scripts and batch runners for study variants).
- Tests: `tests/` (pytest suite; comprehensive coverage of `debug` package).
- Results: `results/` (generated artifacts and plots). Do not hand‑edit.
- Root files: `pyproject.toml` (Python 3.12, deps via `uv`), `README.md` (usage), `CLAUDE.md` (notes).

## Build, Test, and Development Commands
```bash
# Install deps and set up env
uv sync && uv pip install -e .

# Lint and format
ruff check src/ tests/ && ruff format src/ tests/

# Run tests
pytest -q

# Run experiments (examples)
cd experiments && ./run_experiments.sh all
uv run _08_full_layer_token_patching.py --rng-seed 11 --num-hops 1 --seq-len 5
```

## Coding Style & Naming Conventions
- Python style: PEP 8 with 4‑space indents; prefer type hints.
- Imports: standard → third‑party → local; no wildcard imports.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.
- Linting/formatting: use `ruff` (keep diffs clean by running before commits).

## Testing Guidelines
- Framework: `pytest` in `tests/`. Name tests `test_*.py`, functions `test_*`.
- Aim to maintain or improve coverage; add tests alongside new modules.
- Keep tests deterministic (seed RNG with `np.random.RandomState(seed)`).
- Fast unit tests live in `tests/`; long runs belong in `experiments/` and should not be part of CI.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars), e.g., "add causal tracing plot"; group related changes.
- PRs: clear description, rationale, and scope; link issues; include "How to test" steps and, when relevant, sample outputs from `results/` (plots, JSON summaries).
- Ensure `ruff` and `pytest` pass before requesting review.

## Security & Configuration Tips
- Do not commit large model weights or secrets; keep `results/` lightweight.
- Use environment variables for tokens (e.g., Hugging Face); never hardcode.
- Prefer `uv run` for tool entrypoints to ensure the correct environment.

