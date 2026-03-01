"""
Extract random QA pairs from MMLU professional law.
"""

import json
import random
import uuid
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MMLU_PATH = REPO_ROOT / "scripts" / "mmlu_professional_law.json"
DATASET_DIR = REPO_ROOT / "dataset"

# Output sizes
SIZE_10 = 10
SIZE_1000 = 1000


def mmlu_to_medmcqa_format(item: dict, original_index: int) -> dict:
    """Convert one MMLU item to MedMCQA-style format."""
    choices = item["choices"]
    answer_idx = item["answer"]
    letter = chr(ord("A") + answer_idx)
    options = {chr(ord("A") + i): choices[i] for i in range(len(choices))}
    return {
        "question": item["question"],
        "options": options,
        "correct_answer": letter,
        "metadata": {
            "source": "MMLU",
            "id": str(uuid.uuid4()),
            "choice_type": "single",
            "subject_name": item.get("subject", "professional_law"),
            "topic_name": "",
            "original_index": original_index,
        },
    }


def main():
    random.seed(42)

    with open(MMLU_PATH, "r") as f:
        data = json.load(f)

    n_total = len(data)
    if n_total < SIZE_1000:
        raise SystemExit(
            f"Not enough items: have {n_total}, need at least {SIZE_1000}"
        )

    indices = list(range(n_total))
    random.shuffle(indices)

    # Sample 1000 (used for both; 10 is a subset for the small file)
    idx_1000 = indices[:SIZE_1000]
    idx_10 = indices[:SIZE_10]

    out_1000 = [mmlu_to_medmcqa_format(data[i], i) for i in idx_1000]
    out_10 = [mmlu_to_medmcqa_format(data[i], i) for i in idx_10]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    path_1000 = DATASET_DIR / "mmlu_professional_law_1000.json"
    path_10 = DATASET_DIR / "mmlu_professional_law_10.json"

    with open(path_1000, "w") as f:
        json.dump(out_1000, f, indent=2)

    with open(path_10, "w") as f:
        json.dump(out_10, f, indent=2)

    print(f"Wrote {len(out_1000)} items to {path_1000}")
    print(f"Wrote {len(out_10)} items to {path_10}")


if __name__ == "__main__":
    main()
