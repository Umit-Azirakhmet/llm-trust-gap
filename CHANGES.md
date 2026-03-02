# Recent changes (for git push)
# By Jessica Kim, Mar 1st

## 1. `src/experiment.py` – `--only-indices` added

- **`run(..., only_indices=None)`**
  - When `only_indices` is set, only those `question_id`s are processed.
  - Indices are sorted and run in order; progress is logged every 10 items.
- **CLI**
  - Accepts comma-separated indices for re-running a subset, e.g. `--only-indices 47,280,284`.
- **Output file**
  - When `only_indices` is used, the output filename includes `_retry` and a separate retry JSON is saved.

## 2. `scripts/retry_failed.py` (new file)

- **Purpose**
  - Scans result JSONs in `outputs/results_llama`, `results_mistral`, `results_gemma`, `results_qwen`,
  - Collects `question_id`s of failed entries (`"error"` present or `outputted_answer` is null),
  - Re-runs the experiment only for those indices,
  - Merges the new results back into the original JSON files (in-place replace).
- **Usage**
  - `python scripts/retry_failed.py` — retry failed items and merge into originals
  - `python scripts/retry_failed.py --dry-run` — print what would be retried (no API calls)
  - `python scripts/retry_failed.py --no-merge` — run retries only, do not merge
  - Use `--results-dirs` and `--dataset` to customize paths and dataset.

## 3. `src/api_client.py` – timeout increased

- **Change**
  - API request timeout increased from **180s to 360s** (`timeout=360`).
- **Reason**
  - Larger models (e.g. 80B) or server load can exceed 180s; higher timeout reduces spurious failures.

## 4. `scripts/summarize_rq1_all_models.py` (new file)

- **Purpose**
  - Prints **RQ1 (Comparative Analysis by Prompt Group)** for **all four models** in one run.
  - Models: Llama 3.1 8B, Mistral Small, Gemma 3N 4B, Qwen3 80B.
- **Output**
  - One RQ1 table per model (Group, Strategy, Acc., Mean V, Mean P, V-P Gap, p (t-test vs G0), corr(V,P), p (corr)).
- **Dependencies**
  - Uses only json, pathlib, numpy, scipy (no pandas/statsmodels).
- **Usage**
  - `python scripts/summarize_rq1_all_models.py`
  - `python scripts/summarize_rq1_all_models.py --dataset medmcqa_1000`
  - Reads results from `outputs/results_llama`, `outputs/results_mistral`, etc. under project root.
